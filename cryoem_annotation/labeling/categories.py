"""Label category definitions and utilities."""

from typing import List, Dict, Optional, Any

from cryoem_annotation.core.colors import generate_label_colors


# Default categories for cryo-EM particle classification
DEFAULT_LABEL_CATEGORIES = [
    {'name': 'mature', 'key': '1'},
    {'name': 'immature', 'key': '2'},
    {'name': 'indeterminate', 'key': '3'},
    {'name': 'other', 'key': '4'},
    {'name': 'empty', 'key': '5'},
]


class LabelCategories:
    """Manager for label categories with keyboard mapping and colors."""

    def __init__(self, categories: Optional[List[Dict]] = None):
        """
        Initialize with category list or defaults.

        Args:
            categories: List of category dicts with 'name', 'key', and optional 'color'
        """
        self.categories = categories if categories is not None else DEFAULT_LABEL_CATEGORIES
        self._validate_categories()
        self._assign_colors()
        self._build_mappings()

    def _validate_categories(self):
        """Validate category definitions."""
        if not self.categories:
            raise ValueError("At least one category is required")

        names = set()
        keys = set()
        for cat in self.categories:
            if 'name' not in cat:
                raise ValueError("Each category must have a 'name'")
            if 'key' not in cat:
                raise ValueError(f"Category '{cat['name']}' must have a 'key'")
            if cat['name'] in names:
                raise ValueError(f"Duplicate category name: {cat['name']}")
            if cat['key'] in keys:
                raise ValueError(f"Duplicate key binding: {cat['key']}")
            names.add(cat['name'])
            keys.add(cat['key'])

    def _assign_colors(self):
        """Assign colors to categories that don't have them."""
        colors = generate_label_colors(len(self.categories))
        for i, cat in enumerate(self.categories):
            if 'color' not in cat:
                cat['color'] = colors[i]

    def _build_mappings(self):
        """Build lookup dictionaries."""
        self.key_to_name = {cat['key']: cat['name'] for cat in self.categories}
        self.name_to_color = {cat['name']: cat['color'] for cat in self.categories}
        self.name_to_key = {cat['name']: cat['key'] for cat in self.categories}
        self.names = [cat['name'] for cat in self.categories]

    def get_label_for_key(self, key: str) -> Optional[str]:
        """Get label name for a keyboard key."""
        return self.key_to_name.get(key)

    def get_color_for_label(self, label: Any) -> List[float]:
        """Get color for a label (handles both string and legacy int)."""
        if isinstance(label, str):
            return self.name_to_color.get(label, [0.5, 0.5, 0.5])
        elif isinstance(label, int):
            # Legacy integer label: use index-based color
            idx = (label - 1) % len(self.categories) if label > 0 else 0
            return self.categories[idx]['color']
        return [0.5, 0.5, 0.5]

    def get_display_text(self, label: Any) -> str:
        """Get display text for a label."""
        if isinstance(label, str):
            # Use first 3 chars uppercase, e.g., "MAT" for mature
            return label[:3].upper()
        elif isinstance(label, int):
            return f"L{label}"  # Legacy format
        return "?"

    def is_valid_label(self, label: Any) -> bool:
        """Check if label is valid (string category or legacy int)."""
        if isinstance(label, str):
            return label in self.names
        elif isinstance(label, int):
            return True  # Accept legacy integers
        return False

    def get_help_text(self) -> str:
        """Get help text showing key bindings."""
        parts = [f"'{cat['key']}'={cat['name']}" for cat in self.categories]
        return " | ".join(parts)
