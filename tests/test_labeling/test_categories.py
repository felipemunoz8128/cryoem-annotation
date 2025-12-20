"""Tests for label categories module."""

import pytest
from cryoem_annotation.labeling.categories import LabelCategories, DEFAULT_LABEL_CATEGORIES


class TestLabelCategories:
    """Tests for LabelCategories class."""

    def test_default_categories(self):
        """Test that default categories are loaded correctly."""
        cats = LabelCategories()
        assert len(cats.categories) == 5
        assert 'mature' in cats.names
        assert 'immature' in cats.names
        assert 'indeterminate' in cats.names
        assert 'other' in cats.names
        assert 'empty' in cats.names

    def test_key_to_name_mapping(self):
        """Test keyboard key to label name mapping."""
        cats = LabelCategories()
        assert cats.get_label_for_key('1') == 'mature'
        assert cats.get_label_for_key('2') == 'immature'
        assert cats.get_label_for_key('3') == 'indeterminate'
        assert cats.get_label_for_key('4') == 'other'
        assert cats.get_label_for_key('5') == 'empty'
        assert cats.get_label_for_key('9') is None
        assert cats.get_label_for_key('x') is None

    def test_color_assignment(self):
        """Test that colors are assigned to all categories."""
        cats = LabelCategories()
        for name in cats.names:
            color = cats.get_color_for_label(name)
            assert len(color) == 3
            assert all(0 <= c <= 1 for c in color)

    def test_legacy_integer_color(self):
        """Test color lookup for legacy integer labels."""
        cats = LabelCategories()
        color = cats.get_color_for_label(1)
        assert len(color) == 3
        assert all(0 <= c <= 1 for c in color)

    def test_display_text_string_label(self):
        """Test display text for string labels."""
        cats = LabelCategories()
        assert cats.get_display_text('mature') == 'MAT'
        assert cats.get_display_text('immature') == 'IMM'
        assert cats.get_display_text('indeterminate') == 'IND'
        assert cats.get_display_text('other') == 'OTH'
        assert cats.get_display_text('empty') == 'EMP'

    def test_display_text_legacy_integer(self):
        """Test display text for legacy integer labels."""
        cats = LabelCategories()
        assert cats.get_display_text(1) == 'L1'
        assert cats.get_display_text(5) == 'L5'
        assert cats.get_display_text(10) == 'L10'

    def test_display_text_invalid(self):
        """Test display text for invalid labels."""
        cats = LabelCategories()
        assert cats.get_display_text(None) == '?'

    def test_is_valid_label_string(self):
        """Test validation for string labels."""
        cats = LabelCategories()
        assert cats.is_valid_label('mature') is True
        assert cats.is_valid_label('immature') is True
        assert cats.is_valid_label('invalid') is False

    def test_is_valid_label_legacy_integer(self):
        """Test validation for legacy integer labels."""
        cats = LabelCategories()
        assert cats.is_valid_label(1) is True
        assert cats.is_valid_label(99) is True  # Any integer is valid (legacy)

    def test_is_valid_label_invalid_type(self):
        """Test validation for invalid label types."""
        cats = LabelCategories()
        assert cats.is_valid_label(None) is False
        assert cats.is_valid_label([1, 2]) is False

    def test_get_help_text(self):
        """Test help text generation."""
        cats = LabelCategories()
        help_text = cats.get_help_text()
        assert "'1'=mature" in help_text
        assert "'5'=empty" in help_text
        assert "|" in help_text

    def test_custom_categories(self):
        """Test custom category configuration."""
        custom = [
            {'name': 'good', 'key': 'g'},
            {'name': 'bad', 'key': 'b'},
        ]
        cats = LabelCategories(custom)
        assert cats.get_label_for_key('g') == 'good'
        assert cats.get_label_for_key('b') == 'bad'
        assert cats.get_label_for_key('1') is None
        assert len(cats.names) == 2

    def test_custom_categories_with_colors(self):
        """Test custom categories with explicit colors."""
        custom = [
            {'name': 'red_cat', 'key': '1', 'color': [1.0, 0.0, 0.0]},
            {'name': 'blue_cat', 'key': '2', 'color': [0.0, 0.0, 1.0]},
        ]
        cats = LabelCategories(custom)
        assert cats.get_color_for_label('red_cat') == [1.0, 0.0, 0.0]
        assert cats.get_color_for_label('blue_cat') == [0.0, 0.0, 1.0]

    def test_validation_duplicate_name(self):
        """Test that duplicate names are rejected."""
        invalid = [
            {'name': 'test', 'key': '1'},
            {'name': 'test', 'key': '2'},
        ]
        with pytest.raises(ValueError, match="Duplicate category name"):
            LabelCategories(invalid)

    def test_validation_duplicate_key(self):
        """Test that duplicate keys are rejected."""
        invalid = [
            {'name': 'one', 'key': '1'},
            {'name': 'two', 'key': '1'},
        ]
        with pytest.raises(ValueError, match="Duplicate key binding"):
            LabelCategories(invalid)

    def test_validation_missing_name(self):
        """Test that missing name is rejected."""
        invalid = [
            {'key': '1'},
        ]
        with pytest.raises(ValueError, match="must have a 'name'"):
            LabelCategories(invalid)

    def test_validation_missing_key(self):
        """Test that missing key is rejected."""
        invalid = [
            {'name': 'test'},
        ]
        with pytest.raises(ValueError, match="must have a 'key'"):
            LabelCategories(invalid)

    def test_validation_empty_categories(self):
        """Test that empty categories list is rejected."""
        with pytest.raises(ValueError, match="At least one category"):
            LabelCategories([])

    def test_name_to_key_mapping(self):
        """Test reverse mapping from name to key."""
        cats = LabelCategories()
        assert cats.name_to_key['mature'] == '1'
        assert cats.name_to_key['empty'] == '5'
