from __future__ import annotations

from PySide6.QtWidgets import QAbstractItemView, QHeaderView, QListWidget, QTableWidget, QWidget


def enhance_dashboard_widget(widget: QWidget) -> None:
    """Apply advanced terminal defaults to list/table heavy dashboards."""
    for table in widget.findChildren(QTableWidget):
        table.setAlternatingRowColors(True)
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        table.setSortingEnabled(True)
        table.verticalHeader().setVisible(False)
        table.setShowGrid(False)
        table.setWordWrap(False)
        table.setCornerButtonEnabled(False)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        table.horizontalHeader().setHighlightSections(False)
        table.horizontalHeader().setDefaultAlignment(0x84)

    for list_widget in widget.findChildren(QListWidget):
        list_widget.setAlternatingRowColors(True)
        list_widget.setUniformItemSizes(True)
        list_widget.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)

