from __future__ import annotations

import copy
import logging
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Gene:
    name: str
    value: Any
    min_value: float | None = None
    max_value: float | None = None
    gene_type: str = "float"


@dataclass
class Chromosome:
    genes: dict[str, Gene]
    fitness: float = 0.0
    generation: int = 0
    id: str = ""

    def to_params(self) -> dict[str, Any]:
        return {name: gene.value for name, gene in self.genes.items()}

    @classmethod
    def from_params(cls, params: dict[str, Any], generation: int = 0) -> "Chromosome":
        genes = {}
        for name, value in params.items():
            gene_type = "float" if isinstance(value, float) else "int" if isinstance(value, int) else "string"
            genes[name] = Gene(name=name, value=value, gene_type=gene_type)
        chrom_id = f"gen{generation}_{random.randint(1000, 9999)}"
        return cls(genes=genes, generation=generation, id=chrom_id)


class GeneticOptimizer:
    """
    Genetic Algorithm for strategy evolution.
    Implements selection, crossover, mutation, and elitism.
    """

    def __init__(
        self,
        population_size: int = 20,
        elite_size: int = 2,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.7,
        generations: int = 50,
    ):
        self.logger = logging.getLogger(__name__)
        
        self.population_size = population_size
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.generations = generations
        
        self._population: list[Chromosome] = []
        self._best_chromosome: Chromosome | None = None
        self._generation_count = 0
        self._fitness_history: list[float] = []
        
        self._gene_bounds: dict[str, tuple[Any, Any]] = {}
        
    def set_gene_bounds(self, bounds: dict[str, tuple[Any, Any]]) -> None:
        """Set bounds for genes."""
        self._gene_bounds = bounds
        
    def initialize_population(self, base_params: dict[str, Any]) -> None:
        """Initialize random population around base params."""
        
        self._population = []
        
        base_chrom = Chromosome.from_params(base_params, generation=0)
        self._population.append(base_chrom)
        
        for i in range(1, self.population_size):
            chrom_genes = {}
            for name, value in base_params.items():
                min_val, max_val = self._gene_bounds.get(name, (None, None))
                
                if isinstance(value, float):
                    if min_val is not None and max_val is not None:
                        new_value = random.uniform(min_val, max_val)
                    else:
                        variation = abs(value) * 0.3 if value != 0 else 1.0
                        new_value = value + random.gauss(0, variation)
                        new_value = max(0.01, new_value)
                elif isinstance(value, int):
                    if min_val is not None and max_val is not None:
                        new_value = random.randint(min_val, max_val)
                    else:
                        variation = abs(value) * 0.3 if value != 0 else 2
                        new_value = int(value + random.gauss(0, variation))
                        new_value = max(1, new_value)
                else:
                    new_value = value
                
                gene_type = "float" if isinstance(new_value, float) else "int" if isinstance(new_value, int) else "string"
                chrom_genes[name] = Gene(name=name, value=new_value, gene_type=gene_type)
            
            chrom = Chromosome(genes=chrom_genes, generation=0, id=f"gen0_{i}")
            self._population.append(chrom)
        
        self._generation_count = 0
        self.logger.info(f"Initialized population with {len(self._population)} chromosomes")

    def evaluate_population(self, fitness_func: Callable[[dict[str, Any]], float]) -> None:
        """Evaluate fitness for all chromosomes."""
        
        for chrom in self._population:
            params = chrom.to_params()
            try:
                chrom.fitness = fitness_func(params)
            except Exception as e:
                self.logger.error(f"Fitness evaluation error for {params}: {e}")
                chrom.fitness = 0.0
        
        self._population.sort(key=lambda c: c.fitness, reverse=True)
        
        self._best_chromosome = copy.deepcopy(self._population[0])
        
        self._fitness_history.append(self._best_chromosome.fitness)
        
    def selection(self) -> list[Chromosome]:
        """Tournament selection."""
        
        selected = []
        
        for _ in range(self.population_size):
            tournament = random.sample(self._population, k=min(3, len(self._population)))
            winner = max(tournament, key=lambda c: c.fitness)
            selected.append(copy.deepcopy(winner))
        
        return selected

    def crossover(self, parent1: Chromosome, parent2: Chromosome) -> tuple[Chromosome, Chromosome]:
        """Single-point crossover."""
        
        if random.random() > self.crossover_rate:
            return copy.deepcopy(parent1), copy.deepcopy(parent2)
        
        child1_genes = {}
        child2_genes = {}
        
        gene_names = list(parent1.genes.keys())
        if not gene_names:
            return copy.deepcopy(parent1), copy.deepcopy(parent2)
        
        if len(gene_names) <= 1:
            return copy.deepcopy(parent1), copy.deepcopy(parent2)
        
        crossover_point = random.randint(1, len(gene_names) - 1)
        
        for i, name in enumerate(gene_names):
            if i < crossover_point:
                child1_genes[name] = copy.deepcopy(parent1.genes[name])
                child2_genes[name] = copy.deepcopy(parent2.genes[name])
            else:
                child1_genes[name] = copy.deepcopy(parent2.genes[name])
                child2_genes[name] = copy.deepcopy(parent1.genes[name])
        
        child1 = Chromosome(
            genes=child1_genes,
            generation=self._generation_count,
            id=f"gen{self._generation_count}_{random.randint(1000, 9999)}"
        )
        child2 = Chromosome(
            genes=child2_genes,
            generation=self._generation_count,
            id=f"gen{self._generation_count}_{random.randint(1000, 9999)}"
        )
        
        return child1, child2

    def mutate(self, chromosome: Chromosome) -> Chromosome:
        """Gaussian mutation for numeric genes."""
        
        for name, gene in chromosome.genes.items():
            if random.random() > self.mutation_rate:
                continue
                
            min_val, max_val = self._gene_bounds.get(name, (None, None))
            
            if gene.gene_type in ("float", "int"):
                current = gene.value
                if min_val is not None and max_val is not None:
                    range_size = max_val - min_val
                    mutation_strength = range_size * 0.1
                else:
                    mutation_strength = abs(current) * 0.2 if current != 0 else 0.5
                
                mutated = current + random.gauss(0, mutation_strength)
                
                if gene.gene_type == "int":
                    mutated = int(round(mutated))
                
                if min_val is not None:
                    mutated = max(min_val, mutated)
                if max_val is not None:
                    mutated = min(max_val, mutated)
                
                gene.value = mutated
        
        return chromosome

    def evolve(self, fitness_func: Callable[[dict[str, Any]], float]) -> dict[str, Any]:
        """Run one generation of evolution."""
        
        self.evaluate_population(fitness_func)
        
        if not self._population or not self._best_chromosome:
            return self._current_params if hasattr(self, '_current_params') else {}
        
        elites = [copy.deepcopy(c) for c in self._population[:self.elite_size]]
        
        selected = self.selection()
        
        if len(selected) < 2:
            return self._best_chromosome.to_params() if self._best_chromosome else self._current_params
        
        new_population = list(elites)
        
        while len(new_population) < self.population_size:
            if len(selected) >= 2:
                parent1, parent2 = random.sample(selected, 2)
                child1, child2 = self.crossover(parent1, parent2)
                
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                
                new_population.append(child1)
                if len(new_population) < self.population_size:
                    new_population.append(child2)
            else:
                break
        
        self._population = new_population[:self.population_size]
        self._generation_count += 1
        
        best_fit = self._best_chromosome.fitness if self._best_chromosome else 0.0
        self.logger.info(
            f"Generation {self._generation_count}: "
            f"best_fitness={best_fit:.2f}"
        )
        
        return self._best_chromosome.to_params() if self._best_chromosome else self._current_params

    def run(
        self,
        base_params: dict[str, Any],
        fitness_func: Callable[[dict[str, Any]], float],
        min_population_eval: int = 5,
    ) -> dict[str, Any]:
        """Run complete genetic algorithm optimization."""
        
        if self._gene_bounds:
            self.initialize_population(base_params)
        else:
            self._gene_bounds = {
                "rsi_period": (7, 21),
                "rsi_overbought": (60, 80),
                "rsi_oversold": (20, 40),
                "stop_loss_percent": (0.5, 5.0),
                "take_profit_percent": (1.0, 10.0),
                "min_signal_confidence": (0.5, 0.9),
                "position_size_percent": (1.0, 20.0),
                "ma_short_period": (5, 20),
                "ma_long_period": (20, 50),
            }
            self.initialize_population(base_params)
        
        best_overall = None
        best_fitness = float('-inf')
        
        for gen in range(self.generations):
            self.evaluate_population(fitness_func)
            
            if self._best_chromosome.fitness > best_fitness:
                best_fitness = self._best_chromosome.fitness
                best_overall = copy.deepcopy(self._best_chromosome)
            
            if gen < self.generations - 1:
                selected = self.selection()
                new_population = [copy.deepcopy(c) for c in self._population[:self.elite_size]]
                
                while len(new_population) < self.population_size:
                    parent1, parent2 = random.sample(selected, 2)
                    child1, child2 = self.crossover(parent1, parent2)
                    child1 = self.mutate(child1)
                    child2 = self.mutate(child2)
                    new_population.extend([child1, child2])
                
                self._population = new_population[:self.population_size]
                self._generation_count = gen + 1
        
        self.logger.info(
            f"GA optimization complete. Best fitness: {best_fitness:.2f}"
        )
        
        return best_overall.to_params() if best_overall else base_params

    def get_best_params(self) -> dict[str, Any]:
        """Get best parameters found."""
        if self._best_chromosome:
            return self._best_chromosome.to_params()
        return {}

    def get_status(self) -> dict[str, Any]:
        """Get optimizer status."""
        return {
            "generation": self._generation_count,
            "population_size": len(self._population),
            "best_fitness": self._best_chromosome.fitness if self._best_chromosome else 0.0,
            "best_params": self.get_best_params(),
            "fitness_history": self._fitness_history[-10:] if self._fitness_history else [],
            "mutation_rate": self.mutation_rate,
            "crossover_rate": self.crossover_rate,
        }


class StrategyEvolver:
    """
    High-level strategy evolution using genetic algorithms.
    Works with trading performance metrics.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.optimizer = GeneticOptimizer(
            population_size=15,
            elite_size=2,
            mutation_rate=0.15,
            crossover_rate=0.75,
            generations=30,
        )
        
        self._performance_history: list[dict[str, Any]] = []
        self._current_params: dict[str, Any] = {}
        
    def set_current_params(self, params: dict[str, Any]) -> None:
        """Set current strategy parameters."""
        self._current_params = params
        self.logger.info(f"Strategy evolver initialized with params: {params}")

    def _calculate_fitness(self, params: dict[str, Any]) -> float:
        """Calculate fitness from performance metrics."""
        
        if not self._performance_history:
            return 50.0
        
        recent = self._performance_history[-20:]
        if len(recent) < 3:
            return 30.0
        
        wins = sum(1 for p in recent if p.get("pnl", 0) > 0)
        win_rate = wins / len(recent)
        
        total_pnl = sum(p.get("pnl", 0) for p in recent)
        
        avg_win = sum(p.get("pnl", 0) for p in recent if p.get("pnl", 0) > 0) / max(wins, 1)
        avg_loss = sum(abs(p.get("pnl", 0)) for p in recent if p.get("pnl", 0) < 0) / max(len(recent) - wins, 1)
        profit_factor = avg_win / max(avg_loss, 0.01)
        
        sharpe_estimate = profit_factor * win_rate
        
        fitness = (
            win_rate * 40 +
            min(profit_factor, 3.0) / 3.0 * 30 +
            min(sharpe_estimate, 2.0) / 2.0 * 20 +
            min(len(recent) / 20, 1.0) * 10
        )
        
        consecutive_losses = max(p.get("consecutive_losses", 0) for p in recent)
        if consecutive_losses >= 5:
            fitness *= 0.5
        elif consecutive_losses >= 3:
            fitness *= 0.7
        
        return fitness

    def record_performance(self, performance: dict[str, Any]) -> None:
        """Record performance for evolution."""
        
        self._performance_history.append({
            **performance,
            "timestamp": datetime.now(timezone.utc),
        })
        
        if len(self._performance_history) > 100:
            self._performance_history = self._performance_history[-50:]
        
    def evolve_once(self) -> dict[str, Any]:
        """Run one evolution step."""
        
        if not self._current_params:
            self.logger.warning("No current params set for evolution")
            return {}
        
        try:
            if not self.optimizer._population:
                self.optimizer.initialize_population(self._current_params)
            
            best_params = self.optimizer.evolve(self._calculate_fitness)
            self._current_params = best_params
            return best_params
        except Exception as e:
            self.logger.error(f"Evolution error: {e}")
            return self._current_params

    def run_full_evolution(self, min_trades: int = 10) -> dict[str, Any]:
        """Run complete genetic algorithm optimization."""
        
        if len(self._performance_history) < min_trades:
            self.logger.info(f"Not enough trades ({len(self._performance_history)}/{min_trades}) for evolution")
            return self._current_params
        
        try:
            best_params = self.optimizer.run(
                self._current_params,
                self._calculate_fitness,
            )
            self._current_params = best_params
            self.logger.info(f"Full evolution complete. Best params: {best_params}")
            return best_params
        except Exception as e:
            self.logger.error(f"Full evolution error: {e}")
            return self._current_params

    def get_suggested_diversity(self) -> list[dict[str, Any]]:
        """Get diverse parameter suggestions for exploration."""
        
        if not self.optimizer._population:
            return [self._current_params]
        
        sorted_pop = sorted(self.optimizer._population, key=lambda c: c.fitness, reverse=True)
        
        diversity = []
        for chrom in sorted_pop[:5]:
            diversity.append(chrom.to_params())
        
        return diversity

    def get_status(self) -> dict[str, Any]:
        """Get evolver status."""
        return {
            "optimizer": self.optimizer.get_status(),
            "current_params": self._current_params,
            "performance_history_count": len(self._performance_history),
        }
