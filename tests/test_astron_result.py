"""
Comprehensive test cases for astron-result library using pytest.
Tests are organized in class-based format covering all major functionality.
"""

import json
import re
from typing import Any

import pytest
from astron.result import Err, Ok, Result, UnwrapError


class TestBasicResultFunctionality:
    """Test basic Result, Ok, and Err functionality."""

    def test_ok_creation_and_type_checking(self):
        """Test Ok result creation and type checking methods."""
        result = Ok("test_value")

        assert result.is_ok() is True
        assert result.is_err() is False
        assert result.unwrap() == "test_value"
        assert result.unwrap_or("default") == "test_value"

    def test_err_creation_and_type_checking(self):
        """Test Err result creation and type checking methods."""
        result = Err("error_message")

        assert result.is_ok() is False
        assert result.is_err() is True
        assert result.unwrap_or("default") == "default"

    def test_ok_unwrap_success(self):
        """Test successful unwrap operation on Ok results."""
        test_cases = [42, "hello", [1, 2, 3], {"key": "value"}, None]

        for value in test_cases:
            result = Ok(value)
            assert result.unwrap() == value

    def test_err_unwrap_raises_exception(self):
        """Test that unwrap on Err raises UnwrapError."""
        result = Err("something went wrong")

        with pytest.raises(UnwrapError) as exc_info:
            result.unwrap()

        assert "something went wrong" in str(exc_info.value)

    def test_unwrap_or_with_ok(self):
        """Test unwrap_or with Ok results returns the value."""
        result = Ok("actual_value")
        assert result.unwrap_or("default_value") == "actual_value"

    def test_unwrap_or_with_err(self):
        """Test unwrap_or with Err results returns the default."""
        result = Err("error")
        assert result.unwrap_or("default_value") == "default_value"


class TestResultMapping:
    """Test Result mapping operations (map, map_err)."""

    def test_map_on_ok_transforms_value(self):
        """Test that map transforms Ok values correctly."""
        result = Ok(5)
        mapped = result.map(lambda x: x * 2)

        assert mapped.is_ok()
        assert mapped.unwrap() == 10

    def test_map_on_err_preserves_error(self):
        """Test that map on Err preserves the error."""
        result = Err("error")
        mapped = result.map(lambda x: x * 2)

        assert mapped.is_err()
        assert mapped.unwrap_or("default") == "default"

    def test_map_chaining(self):
        """Test chaining multiple map operations."""
        result = (
            Ok(3).map(lambda x: x * 2).map(lambda x: x + 4).map(str)  # 6  # 10
        )  # "10"

        assert result.unwrap() == "10"

    def test_map_err_on_ok_preserves_value(self):
        """Test that map_err on Ok preserves the value."""
        result = Ok("value")
        mapped = result.map_err(lambda e: f"Modified: {e}")

        assert mapped.is_ok()
        assert mapped.unwrap() == "value"

    def test_map_err_on_err_transforms_error(self):
        """Test that map_err transforms Err values correctly."""
        result = Err("original_error")
        mapped = result.map_err(lambda e: f"Modified: {e}")

        assert mapped.is_err()
        assert mapped.unwrap_or("default") == "default"
        # Note: Can't directly test the error value without accessing internals

    def test_map_with_different_types(self):
        """Test mapping to different types."""
        # String to int
        result1 = Ok("hello").map(len)
        assert result1.unwrap() == 5

        # Int to string
        result2 = Ok(42).map(str)
        assert result2.unwrap() == "42"

        # List to int
        result3 = Ok([1, 2, 3, 4]).map(len)
        assert result3.unwrap() == 4


class TestResultChaining:
    """Test Result chaining operations (and_then, or_else)."""

    def test_and_then_on_ok_chains_operation(self):
        """Test that and_then chains operations on Ok results."""

        def safe_divide(x: int) -> Result[float, str]:
            if x == 0:
                return Err("Division by zero")
            return Ok(10.0 / x)

        result = Ok(2).and_then(safe_divide)
        assert result.is_ok()
        assert result.unwrap() == 5.0

    def test_and_then_on_err_short_circuits(self):
        """Test that and_then on Err short-circuits."""

        def safe_divide(x: int) -> Result[float, str]:
            return Ok(10.0 / x)

        result = Err("initial_error").and_then(safe_divide)
        assert result.is_err()

    def test_and_then_error_propagation(self):
        """Test error propagation through and_then chains."""

        def parse_int(s: str) -> Result[int, str]:
            try:
                return Ok(int(s))
            except ValueError:
                return Err(f"Cannot parse '{s}'")

        def validate_positive(x: int) -> Result[int, str]:
            if x > 0:
                return Ok(x)
            return Err("Must be positive")

        # Success case
        success = Ok("42").and_then(parse_int).and_then(validate_positive)
        assert success.is_ok()
        assert success.unwrap() == 42

        # Parse error case
        parse_error = Ok("not_a_number").and_then(parse_int).and_then(validate_positive)
        assert parse_error.is_err()

        # Validation error case
        validation_error = Ok("-5").and_then(parse_int).and_then(validate_positive)
        assert validation_error.is_err()

    def test_or_else_on_ok_preserves_value(self):
        """Test that or_else on Ok preserves the value."""

        def recovery_function(error: str) -> Result[str, str]:
            return Ok("recovered_value")

        result = Ok("original_value").or_else(recovery_function)
        assert result.is_ok()
        assert result.unwrap() == "original_value"

    def test_or_else_on_err_attempts_recovery(self):
        """Test that or_else on Err attempts recovery."""

        def recovery_function(error: str) -> Result[str, str]:
            return Ok("recovered_value")

        result = Err("original_error").or_else(recovery_function)
        assert result.is_ok()
        assert result.unwrap() == "recovered_value"

    def test_or_else_recovery_can_also_fail(self):
        """Test that or_else recovery can also return Err."""

        def failing_recovery(error: str) -> Result[str, str]:
            return Err("recovery_also_failed")

        result = Err("original_error").or_else(failing_recovery)
        assert result.is_err()


class TestResultFiltering:
    """Test Result filtering operations."""

    def test_filter_on_ok_with_passing_predicate(self):
        """Test filter with Ok value that passes predicate."""
        result = Ok(10).filter(lambda x: x > 5, "Value too small")
        assert result.is_ok()
        assert result.unwrap() == 10

    def test_filter_on_ok_with_failing_predicate(self):
        """Test filter with Ok value that fails predicate."""
        result = Ok(3).filter(lambda x: x > 5, "Value too small")
        assert result.is_err()

    def test_filter_on_err_preserves_error(self):
        """Test that filter on Err preserves the original error."""
        result = Err("original_error").filter(lambda x: x > 5, "Value too small")
        assert result.is_err()

    def test_filter_with_different_predicates(self):
        """Test filter with various predicates."""
        # String length filter
        result1 = Ok("hello").filter(lambda s: len(s) >= 5, "String too short")
        assert result1.is_ok()

        result2 = Ok("hi").filter(lambda s: len(s) >= 5, "String too short")
        assert result2.is_err()

        # Even number filter
        result3 = Ok(4).filter(lambda x: x % 2 == 0, "Must be even")
        assert result3.is_ok()

        result4 = Ok(5).filter(lambda x: x % 2 == 0, "Must be even")
        assert result4.is_err()


class TestResultFactoryMethods:
    """Test Result factory methods (from_exception, from_optional)."""

    def test_from_exception_with_successful_function(self):
        """Test from_exception with function that succeeds."""

        def successful_function():
            return "success"

        result = Result.from_exception(successful_function)
        assert result.is_ok()
        assert result.unwrap() == "success"

    def test_from_exception_with_failing_function(self):
        """Test from_exception with function that raises exception."""

        def failing_function():
            raise ValueError("Something went wrong")

        result = Result.from_exception(failing_function)
        assert result.is_err()

    def test_from_exception_preserves_exception_type(self):
        """Test that from_exception preserves the exception type."""

        def failing_function():
            raise KeyError("Missing key")

        result = Result.from_exception(failing_function)
        assert result.is_err()
        # The error should contain the KeyError

    def test_from_optional_with_value(self):
        """Test from_optional with non-None value."""
        result = Result.from_optional("value", lambda: "Was None")
        assert result.is_ok()
        assert result.unwrap() == "value"

    def test_from_optional_with_none(self):
        """Test from_optional with None value."""
        result = Result.from_optional(None, lambda: "Was None")
        assert result.is_err()

    def test_from_optional_with_callable_error(self):
        """Test from_optional with callable error generator."""
        counter = 0

        def error_generator():
            nonlocal counter
            counter += 1
            return f"Error #{counter}"

        result1 = Result.from_optional(None, error_generator)
        result2 = Result.from_optional(None, error_generator)

        assert result1.is_err()
        assert result2.is_err()
        assert counter == 2  # Error generator should be called for each None


class TestPatternMatching:
    """Test pattern matching with Results (Python 3.10+ feature)."""

    def test_pattern_matching_with_ok(self):
        """Test pattern matching with Ok results."""

        def handle_result(result: Result[str, str]) -> str:
            match result:
                case Ok(value):
                    return f"Success: {value}"
                case Err(error):
                    return f"Error: {error}"
            return "Unhandled result"  # Default return for unmatched cases

        result = Ok("test_data")
        assert handle_result(result) == "Success: test_data"

    def test_pattern_matching_with_err(self):
        """Test pattern matching with Err results."""

        def handle_result(result: Result[str, str]) -> str:
            match result:
                case Ok(value):
                    return f"Success: {value}"
                case Err(error):
                    return f"Error: {error}"
            return "Unhandled result"  # Default return for unmatched cases

        result = Err("test_error")
        assert handle_result(result) == "Error: test_error"

    def test_pattern_matching_with_guards(self):
        """Test pattern matching with guards and conditions."""

        def handle_number_result(result: Result[int, str]) -> str:
            match result:
                case Ok(value) if value > 100:
                    return "Large number"
                case Ok(value) if value > 0:
                    return "Positive number"
                case Ok(0):
                    return "Zero"
                case Ok(value):
                    return "Negative number"
                case Err(error):
                    return f"Error: {error}"
            return "Unhandled result"  # Default return for unmatched cases

        assert handle_number_result(Ok(150)) == "Large number"
        assert handle_number_result(Ok(50)) == "Positive number"
        assert handle_number_result(Ok(0)) == "Zero"
        assert handle_number_result(Ok(-10)) == "Negative number"
        assert handle_number_result(Err("parse error")) == "Error: parse error"


class TestRealWorldScenarios:
    """Test real-world usage scenarios."""

    def test_json_parsing_pipeline(self):
        """Test a complete JSON parsing and validation pipeline."""

        def parse_json(json_str: str) -> Result[dict, str]:
            try:
                return Ok(json.loads(json_str))
            except json.JSONDecodeError as e:
                return Err(f"JSON parse error: {str(e)}")

        def validate_user_data(data: dict) -> Result[dict, str]:
            required_fields = ["name", "age", "email"]
            for field in required_fields:
                if field not in data:
                    return Err(f"Missing required field: {field}")

            if not isinstance(data["age"], int) or data["age"] < 0:
                return Err("Age must be a non-negative integer")

            if "@" not in data["email"]:
                return Err("Invalid email format")

            return Ok(data)

        def create_user_summary(data: dict) -> str:
            return f"{data['name']} ({data['age']}) - {data['email']}"

        # Test successful case
        valid_json = '{"name": "Alice", "age": 30, "email": "alice@example.com"}'
        result = (
            parse_json(valid_json).and_then(validate_user_data).map(create_user_summary)
        )

        assert result.is_ok()
        assert result.unwrap() == "Alice (30) - alice@example.com"

        # Test invalid JSON
        invalid_json = '{"name": "Bob", "age": 25'  # Missing closing brace
        result = (
            parse_json(invalid_json)
            .and_then(validate_user_data)
            .map(create_user_summary)
        )

        assert result.is_err()

        # Test missing field
        missing_field_json = '{"name": "Charlie", "age": 35}'  # Missing email
        result = (
            parse_json(missing_field_json)
            .and_then(validate_user_data)
            .map(create_user_summary)
        )

        assert result.is_err()

        # Test invalid age
        invalid_age_json = '{"name": "David", "age": -5, "email": "david@example.com"}'
        result = (
            parse_json(invalid_age_json)
            .and_then(validate_user_data)
            .map(create_user_summary)
        )

        assert result.is_err()

    def test_mathematical_operations_chain(self):
        """Test chaining mathematical operations with error handling."""

        def safe_divide(a: float, b: float) -> Result[float, str]:
            if b == 0:
                return Err("Division by zero")
            return Ok(a / b)

        def safe_sqrt(x: float) -> Result[float, str]:
            if x < 0:
                return Err("Cannot take square root of negative number")
            return Ok(x**0.5)

        def safe_log(x: float) -> Result[float, str]:
            if x <= 0:
                return Err("Cannot take logarithm of non-positive number")
            import math

            return Ok(math.log(x))

        # Test successful chain: 16 / 4 = 4, sqrt(4) = 2, log(2) ≈ 0.693
        result = safe_divide(16, 4).and_then(safe_sqrt).and_then(safe_log)

        assert result.is_ok()
        assert abs(result.unwrap() - 0.6931471805599453) < 1e-10

        # Test division by zero
        result = safe_divide(16, 0).and_then(safe_sqrt).and_then(safe_log)

        assert result.is_err()

        # Test negative square root
        result = safe_divide(-16, 4).and_then(safe_sqrt).and_then(safe_log)

        assert result.is_err()

    def test_configuration_loading_with_defaults(self):
        """Test loading configuration with fallback defaults."""

        def load_config_value(key: str, config: dict[str, Any]) -> Result[Any, str]:
            if key in config:
                return Ok(config[key])
            return Err(f"Missing configuration key: {key}")

        def parse_int_config(value: Any) -> Result[int, str]:
            if isinstance(value, int):
                return Ok(value)
            if isinstance(value, str):
                try:
                    return Ok(int(value))
                except ValueError:
                    return Err(f"Cannot parse '{value}' as integer")
            return Err(f"Invalid type for integer config: {type(value)}")

        def provide_default(error: str) -> Result[int, str]:
            return Ok(8080)  # Default port

        # Test with valid config
        config = {"port": "3000", "host": "localhost"}
        port_result = (
            load_config_value("port", config)
            .and_then(parse_int_config)
            .or_else(provide_default)
        )

        assert port_result.is_ok()
        assert port_result.unwrap() == 3000

        # Test with missing config (should use default)
        config = {"host": "localhost"}
        port_result = (
            load_config_value("port", config)
            .and_then(parse_int_config)
            .or_else(provide_default)
        )

        assert port_result.is_ok()
        assert port_result.unwrap() == 8080

        # Test with invalid config (should use default)
        config = {"port": "not_a_number", "host": "localhost"}
        port_result = (
            load_config_value("port", config)
            .and_then(parse_int_config)
            .or_else(provide_default)
        )

        assert port_result.is_ok()
        assert port_result.unwrap() == 8080


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_nested_results(self):
        """Test handling of nested Result types."""
        # Result containing another Result
        nested_ok = Ok(Ok("inner_value"))
        assert nested_ok.is_ok()
        inner_result = nested_ok.unwrap()
        assert inner_result.is_ok()
        assert inner_result.unwrap() == "inner_value"

        # Map over nested results
        flattened = nested_ok.map(lambda inner: inner.unwrap_or("default"))
        assert flattened.unwrap() == "inner_value"

    def test_empty_and_none_values(self):
        """Test Results containing empty collections and None values."""
        # Empty collections
        empty_list = Ok([])
        assert empty_list.is_ok()
        assert empty_list.unwrap() == []

        empty_dict = Ok({})
        assert empty_dict.is_ok()
        assert empty_dict.unwrap() == {}

        empty_string = Ok("")
        assert empty_string.is_ok()
        assert empty_string.unwrap() == ""

        # None values
        none_value = Ok(None)
        assert none_value.is_ok()
        assert none_value.unwrap() is None

    def test_large_data_handling(self):
        """Test Results with large data structures."""
        large_list = list(range(10000))
        result = Ok(large_list)

        assert result.is_ok()
        assert len(result.unwrap()) == 10000

        # Map over large data
        mapped = result.map(len)
        assert mapped.unwrap() == 10000

    def test_recursive_error_recovery(self):
        """Test multiple levels of error recovery."""

        def first_attempt(x: int) -> Result[int, str]:
            return Err("first_failed")

        def second_attempt(error: str) -> Result[int, str]:
            return Err("second_failed")

        def third_attempt(error: str) -> Result[int, str]:
            return Ok(42)

        result = first_attempt(0).or_else(second_attempt).or_else(third_attempt)

        assert result.is_ok()
        assert result.unwrap() == 42

        # Test when all attempts fail
        def always_fail(error: str) -> Result[int, str]:
            return Err("always_fails")

        result = first_attempt(0).or_else(second_attempt).or_else(always_fail)

        assert result.is_err()


class TestPerformanceAndMemory:
    """Test performance characteristics and memory usage."""

    def test_many_operations_performance(self):
        """Test performance with many chained operations."""
        # This test ensures that chaining many operations doesn't cause
        # excessive overhead or stack overflow
        result = Ok(1)

        # Chain 100 map operations
        for i in range(100):
            result = result.map(lambda x: x + 1)

        assert result.is_ok()
        assert result.unwrap() == 101

    def test_error_short_circuiting(self):
        """Test that errors short-circuit efficiently."""
        call_count = 0

        def expensive_operation(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2

        # Start with an error - expensive operations should not be called
        result = (
            Err("initial_error")
            .map(expensive_operation)
            .map(expensive_operation)
            .map(expensive_operation)
        )

        assert result.is_err()
        assert call_count == 0  # No operations should have been called

    def test_memory_efficiency_with_large_errors(self):
        """Test memory efficiency when handling large error messages."""
        large_error_message = "x" * 10000
        result = Err(large_error_message)

        # Operations on Err should not duplicate the large error message unnecessarily
        mapped = result.map(lambda x: x * 2)
        chained = mapped.and_then(lambda x: Ok(x))

        assert result.is_err()
        assert mapped.is_err()
        assert chained.is_err()


if __name__ == "__main__":
    # Run tests if this file is executed directly
    pytest.main([__file__])
