from __future__ import annotations

import logging
import random
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class StrategyGene:
    gene_type: str
    value: float
    mutation_rate: float = 0.1


@dataclass
class EvolvedStrategy:
    strategy_id: str
    genes: dict[str, StrategyGene]
    fitness: float = 0.0
    generation: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    performance_history: list[float] = field(default_factory=list)


class EnhancedGeneticOptimizer:
    def __init__(self, population_size: int = 30, elite_size: int = 3):
        self.logger = logging.getLogger(__name__)
        
        self.population_size = population_size
        self.elite_size = elite_size
        
        self._population: list[EvolvedStrategy] = []
        self._generation = 0
        self._best_strategy: EvolvedStrategy | None = None
        
        self._history: list[dict] = []
        
        self._initialize_population()
        
        self.logger.info(f"Enhanced Genetic Optimizer initialized with {population_size} strategies")

    def _initialize_population(self) -> None:
        param_ranges = {
            "rsi_period": (10, 30),
            "rsi_oversold": (20, 40),
            "rsi_overbought": (60, 80),
            "stop_loss": (0.01, 0.05),
            "take_profit": (0.02, 0.10),
            "position_size": (0.05, 0.20),
            "ma_short": (5, 20),
            "ma_long": (20, 50),
            "atr_multiplier": (1.0, 3.0),
        }
        
        for i in range(self.population_size):
            genes = {}
            
            for param, (min_val, max_val) in param_ranges.items():
                genes[param] = StrategyGene(
                    gene_type=param,
                    value=random.uniform(min_val, max_val),
                    mutation_rate=0.1
                )
            
            strategy = EvolvedStrategy(
                strategy_id=f"gen_{self._generation}_strat_{i}",
                genes=genes,
                generation=self._generation
            )
            
            self._population.append(strategy)

    def _calculate_fitness(self, strategy: EvolvedStrategy, metrics: dict) -> float:
        win_rate = metrics.get("win_rate", 0.5)
        profit_factor = metrics.get("profit_factor", 1.0)
        sharpe = metrics.get("sharpe", 0.0)
        trade_count = metrics.get("trade_count", 10)
        
        fitness = (
            win_rate * 40 +
            profit_factor * 30 +
            sharpe * 20 +
            min(trade_count / 100, 1.0) * 10
        )
        
        consecutive_losses = metrics.get("consecutive_losses", 0)
        if consecutive_losses >= 5:
            fitness *= 0.5
        
        return fitness

    def evolve(self, performance_metrics: dict[str, dict]) -> dict:
        self._generation += 1
        
        for strategy in self._population:
            symbol = list(performance_metrics.keys())[0] if performance_metrics else "default"
            metrics = performance_metrics.get(symbol, {})
            
            strategy.fitness = self._calculate_fitness(strategy, metrics)
            strategy.performance_history.append(strategy.fitness)
        
        self._population.sort(key=lambda s: s.fitness, reverse=True)
        
        self._best_strategy = self._population[0]
        
        elite = self._population[:self.elite_size]
        
        new_population = [s for s in elite]
        
        while len(new_population) < self.population_size:
            parent1 = self._tournament_select()
            parent2 = self._tournament_select()
            
            child = self._crossover(parent1, parent2)
            
            child = self._mutate(child)
            
            new_population.append(child)
        
        self._population = new_population
        
        self._history.append({
            "generation": self._generation,
            "best_fitness": self._best_strategy.fitness,
            "avg_fitness": sum(s.fitness for s in self._population) / len(self._population),
            "timestamp": datetime.now()
        })
        
        return {
            "generation": self._generation,
            "best_fitness": self._best_strategy.fitness,
            "best_genes": {k: v.value for k, v in self._best_strategy.genes.items()}
        }

    def _tournament_select(self) -> EvolvedStrategy:
        tournament = random.sample(self._population, k=min(3, len(self._population)))
        return max(tournament, key=lambda s: s.fitness)

    def _crossover(self, parent1: EvolvedStrategy, parent2: EvolvedStrategy) -> EvolvedStrategy:
        child_genes = {}
        
        for gene_name in parent1.genes:
            if random.random() < 0.5:
                child_genes[gene_name] = StrategyGene(
                    gene_type=gene_name,
                    value=parent1.genes[gene_name].value,
                    mutation_rate=parent1.genes[gene_name].mutation_rate
                )
            else:
                child_genes[gene_name] = StrategyGene(
                    gene_type=gene_name,
                    value=parent2.genes[gene_name].value,
                    mutation_rate=parent2.genes[gene_name].mutation_rate
                )
        
        child = EvolvedStrategy(
            strategy_id=f"gen_{self._generation + 1}_child_{random.randint(0, 9999)}",
            genes=child_genes,
            generation=self._generation + 1
        )
        
        return child

    def _mutate(self, strategy: EvolvedStrategy) -> EvolvedStrategy:
        for gene_name, gene in strategy.genes.items():
            if random.random() < gene.mutation_rate:
                mutation_strength = 0.2
                
                delta = random.uniform(-1, 1) * mutation_strength * gene.value
                gene.value += delta
                
                gene.value = max(0.001, gene.value)
        
        return strategy

    def get_best_strategy(self) -> dict:
        if not self._best_strategy:
            return {}
        
        return {
            "strategy_id": self._best_strategy.strategy_id,
            "fitness": self._best_strategy.fitness,
            "generation": self._best_strategy.generation,
            "genes": {k: v.value for k, v in self._best_strategy.genes.items()},
            "performance_history": self._best_strategy.performance_history
        }

    def get_optimizer_stats(self) -> dict:
        return {
            "generation": self._generation,
            "population_size": len(self._population),
            "best_fitness": self._best_strategy.fitness if self._best_strategy else 0,
            "avg_fitness": sum(s.fitness for s in self._population) / len(self._population) if self._population else 0,
            "elite_fitness": sum(s.fitness for s in self._population[:self.elite_size]) / self.elite_size if self._population else 0
        }


class NoveltySearchOptimizer:
    def __init__(self, population_size: int = 20, novelty_threshold: float = 0.3):
        self.logger = logging.getLogger(__name__)
        
        self.population_size = population_size
        self.novelty_threshold = novelty_threshold
        
        self._population: list[EvolvedStrategy] = []
        self._archive: list[EvolvedStrategy] = []
        
        self._initialize_population()
        
        self.logger.info("NoveltySearchOptimizer initialized")

    def _initialize_population(self) -> None:
        param_ranges = {
            "rsi_period": (10, 30),
            "stop_loss": (0.01, 0.05),
            "take_profit": (0.02, 0.10),
            "position_size": (0.05, 0.20),
        }
        
        for i in range(self.population_size):
            genes = {}
            for param, (min_val, max_val) in param_ranges.items():
                genes[param] = StrategyGene(
                    gene_type=param,
                    value=random.uniform(min_val, max_val),
                    mutation_rate=0.15
                )
            
            strategy = EvolvedStrategy(
                strategy_id=f"novelty_{i}",
                genes=genes
            )
            self._population.append(strategy)

    def _calculate_novelty(self, strategy: EvolvedStrategy) -> float:
        all_strategies = self._population + self._archive
        
        if not all_strategies:
            return 1.0
        
        distances = []
        for other in all_strategies:
            if other.strategy_id == strategy.strategy_id:
                continue
            
            dist = sum(
                abs(strategy.genes[k].value - other.genes[k].value)
                for k in strategy.genes
                if k in other.genes
            )
            distances.append(dist)
        
        if not distances:
            return 1.0
        
        return sum(distances) / len(distances)

    def _select_with_novelty(self, performance_metrics: dict) -> list[EvolvedStrategy]:
        for strategy in self._population:
            symbol = list(performance_metrics.keys())[0] if performance_metrics else "default"
            metrics = performance_metrics.get(symbol, {"win_rate": 0.5})
            strategy.fitness = metrics.get("win_rate", 0.5)
        
        fitness_scores = [s.fitness for s in self._population]
        novelty_scores = [self._calculate_novelty(s) for s in self._population]
        
        combined_scores = []
        for i, strategy in enumerate(self._population):
            fitness_norm = (fitness_scores[i] - min(fitness_scores)) / (max(fitness_scores) - min(fitness_scores) + 0.001)
            novelty_norm = novelty_scores[i] / max(novelty_scores)
            
            combined = 0.5 * fitness_norm + 0.5 * novelty_norm
            combined_scores.append((strategy, combined))
        
        combined_scores.sort(key=lambda x: x[1], reverse=True)
        
        return [s[0] for s in combined_scores[:self.population_size]]

    def evolve(self, performance_metrics: dict) -> dict:
        selected = self._select_with_novelty(performance_metrics)
        
        self._archive.extend(self._population[:2])
        
        if len(self._archive) > 50:
            self._archive = self._archive[-50:]
        
        self._population = selected
        
        best = max(self._population, key=lambda s: s.fitness)
        
        return {
            "best_fitness": best.fitness,
            "best_genes": {k: v.value for k, v in best.genes.items()},
            "archive_size": len(self._archive),
            "novelty_scores": [self._calculate_novelty(s) for s in self._population[:5]]
        }


class StrategyEvolutionManager:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        self._genetic_optimizer = EnhancedGeneticOptimizer()
        self._novelty_search = NoveltySearchOptimizer()
        
        self._use_novelty_search = False
        self._stagnation_counter = 0
        
        self._evolution_history: list[dict] = []
        
        self.logger.info("StrategyEvolutionManager initialized")

    def evolve(self, performance_metrics: dict) -> dict:
        if self._stagnation_counter >= 5:
            self.logger.info("Activating Novelty Search due to stagnation")
            self._use_novelty_search = True
            self._stagnation_counter = 0
        
        if self._use_novelty_search:
            result = self._novelty_search.evolve(performance_metrics)
        else:
            result = self._genetic_optimizer.evolve(performance_metrics)
        
        self._evolution_history.append(result)
        
        if result.get("best_fitness", 0) <= 0:
            self._stagnation_counter += 1
        else:
            self._stagnation_counter = 0
        
        if len(self._evolution_history) > 20:
            recent = self._evolution_history[-10:]
            fitnesses = [r.get("best_fitness", 0) for r in recent]
            if max(fitnesses) - min(fitnesses) < 1.0:
                self._use_novelty_search = True
        
        return result

    def get_best_strategy(self) -> dict:
        return self._genetic_optimizer.get_best_strategy()

    def get_evolution_stats(self) -> dict:
        return {
            "genetic": self._genetic_optimizer.get_optimizer_stats(),
            "use_novelty_search": self._use_novelty_search,
            "stagnation_counter": self._stagnation_counter,
            "total_evolutions": len(self._evolution_history)
        }