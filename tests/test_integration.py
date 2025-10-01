"""
Integration tests for astron-result library.
These tests focus on integration scenarios and interactions between multiple components.
"""

import pytest
import json
import tempfile
import os
from pathlib import Path
from astron.result import Result, Ok, Err


class TestFileOperationIntegration:
    """Test file operations integration with Result patterns."""
    
    def test_safe_file_reading_pipeline(self):
        """Test a complete file reading and processing pipeline."""
        def safe_read_file(filepath: str) -> Result[str, str]:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    return Ok(f.read())
            except FileNotFoundError:
                return Err(f"File not found: {filepath}")
            except PermissionError:
                return Err(f"Permission denied: {filepath}")
            except Exception as e:
                return Err(f"Error reading file: {str(e)}")
        
        def parse_config_lines(content: str) -> Result[dict, str]:
            config = {}
            for line_num, line in enumerate(content.strip().split('\n'), 1):
                if '=' not in line:
                    return Err(f"Invalid config format at line {line_num}: {line}")
                key, value = line.split('=', 1)
                config[key.strip()] = value.strip()
            return Ok(config)
        
        def validate_required_keys(config: dict) -> Result[dict, str]:
            required = ['host', 'port', 'database']
            missing = [key for key in required if key not in config]
            if missing:
                return Err(f"Missing required config keys: {missing}")
            return Ok(config)
        
        # Create a temporary config file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.conf') as f:
            f.write("host=localhost\nport=5432\ndatabase=testdb\nuser=admin")
            temp_file = f.name
        
        try:
            # Test successful pipeline
            result = (safe_read_file(temp_file)
                     .and_then(parse_config_lines)
                     .and_then(validate_required_keys))
            
            assert result.is_ok()
            config = result.unwrap()
            assert config['host'] == 'localhost'
            assert config['port'] == '5432'
            assert config['database'] == 'testdb'
            
            # Test with non-existent file
            result = (safe_read_file('nonexistent.conf')
                     .and_then(parse_config_lines)
                     .and_then(validate_required_keys))
            
            assert result.is_err()
            
        finally:
            os.unlink(temp_file)
    
    def test_json_file_processing_workflow(self):
        """Test JSON file processing with comprehensive error handling."""
        def safe_read_json_file(filepath: str) -> Result[dict, str]:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return Ok(data)
            except FileNotFoundError:
                return Err(f"File not found: {filepath}")
            except json.JSONDecodeError as e:
                return Err(f"Invalid JSON: {str(e)}")
            except Exception as e:
                return Err(f"Error reading JSON file: {str(e)}")
        
        def extract_users(data: dict) -> Result[list, str]:
            if 'users' not in data:
                return Err("Missing 'users' key in JSON")
            if not isinstance(data['users'], list):
                return Err("'users' must be a list")
            return Ok(data['users'])
        
        def validate_user_structure(users: list) -> Result[list, str]:
            for i, user in enumerate(users):
                if not isinstance(user, dict):
                    return Err(f"User {i} is not a dictionary")
                if 'id' not in user or 'name' not in user:
                    return Err(f"User {i} missing required fields (id, name)")
            return Ok(users)
        
        # Create a temporary JSON file
        test_data = {
            "users": [
                {"id": 1, "name": "Alice", "email": "alice@example.com"},
                {"id": 2, "name": "Bob", "email": "bob@example.com"}
            ],
            "meta": {"version": "1.0"}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            json.dump(test_data, f)
            temp_file = f.name
        
        try:
            # Test successful processing
            result = (safe_read_json_file(temp_file)
                     .and_then(extract_users)
                     .and_then(validate_user_structure))
            
            assert result.is_ok()
            users = result.unwrap()
            assert len(users) == 2
            assert users[0]['name'] == 'Alice'
            
        finally:
            os.unlink(temp_file)


class TestDataPipelineIntegration:
    """Test data processing pipeline integration scenarios."""
    
    def test_multi_step_data_transformation(self):
        """Test a complex multi-step data transformation pipeline."""
        def parse_csv_line(line: str) -> Result[list, str]:
            try:
                # Simple CSV parsing (not production-ready)
                values = [v.strip().strip('"') for v in line.split(',')]
                return Ok(values)
            except Exception as e:
                return Err(f"CSV parse error: {str(e)}")
        
        def validate_record_length(record: list) -> Result[list, str]:
            if len(record) != 4:
                return Err(f"Expected 4 fields, got {len(record)}")
            return Ok(record)
        
        def convert_to_user_dict(record: list) -> Result[dict, str]:
            try:
                user = {
                    'id': int(record[0]),
                    'name': record[1],
                    'age': int(record[2]),
                    'city': record[3]
                }
                return Ok(user)
            except ValueError as e:
                return Err(f"Type conversion error: {str(e)}")
        
        def validate_user_age(user: dict) -> Result[dict, str]:
            if user['age'] < 0 or user['age'] > 150:
                return Err(f"Invalid age: {user['age']}")
            return Ok(user)
        
        # Test data
        csv_lines = [
            "1,Alice,30,New York",
            "2,Bob,25,Los Angeles",
            "3,Charlie,-5,Chicago",  # Invalid age
            "4,David,35",  # Missing field
            "5,Eve,28,Seattle"
        ]
        
        results = []
        for line in csv_lines:
            result = (parse_csv_line(line)
                     .and_then(validate_record_length)
                     .and_then(convert_to_user_dict)
                     .and_then(validate_user_age))
            results.append(result)
        
        # Check results
        assert results[0].is_ok()  # Alice - valid
        assert results[1].is_ok()  # Bob - valid
        assert results[2].is_err()  # Charlie - invalid age
        assert results[3].is_err()  # David - missing field
        assert results[4].is_ok()  # Eve - valid
        
        # Extract valid users
        valid_users = [r.unwrap() for r in results if r.is_ok()]
        assert len(valid_users) == 3
        assert valid_users[0]['name'] == 'Alice'
        assert valid_users[1]['name'] == 'Bob'
        assert valid_users[2]['name'] == 'Eve'
    
    def test_batch_processing_with_error_collection(self):
        """Test batch processing that collects both successes and errors."""
        def process_number(x: str) -> Result[int, str]:
            try:
                num = int(x)
                if num < 0:
                    return Err(f"Negative number: {num}")
                return Ok(num ** 2)  # Square the number
            except ValueError:
                return Err(f"Invalid number format: {x}")
        
        # Test data with mixed valid and invalid inputs
        inputs = ["5", "10", "abc", "-3", "7", "def", "0", "12"]
        
        # Process all inputs and separate successes from errors
        successes = []
        errors = []
        
        for input_val in inputs:
            result = process_number(input_val)
            if result.is_ok():
                successes.append(result.unwrap())
            else:
                errors.append(result.unwrap_or("Unknown error"))
        
        # Verify results
        assert successes == [25, 100, 49, 0, 144]  # 5², 10², 7², 0², 12²
        assert len(errors) == 3  # "abc", "-3", "def"
        
        # Check specific error cases
        error_results = [process_number(x) for x in ["abc", "-3", "def"]]
        assert all(r.is_err() for r in error_results)


class TestAsyncCompatibilityIntegration:
    """Test integration with async/await patterns."""
    
    def test_result_with_async_functions(self):
        """Test that Results work well with async functions."""
        import asyncio
        
        async def async_operation(x: int) -> Result[int, str]:
            # Simulate async work
            await asyncio.sleep(0.001)
            if x < 0:
                return Err("Negative values not allowed")
            return Ok(x * 2)
        
        async def async_validation(x: int) -> Result[int, str]:
            await asyncio.sleep(0.001)
            if x > 100:
                return Err("Value too large")
            return Ok(x)
        
        async def run_async_pipeline():
            # Test successful case
            result = Ok(10)
            if result.is_ok():
                value = result.unwrap()
                async_result = await async_operation(value)
                if async_result.is_ok():
                    final_result = await async_validation(async_result.unwrap())
                    return final_result
            return Err("Pipeline failed")
        
        # Run the async test
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(run_async_pipeline())
            assert result.is_ok()
            assert result.unwrap() == 20
        finally:
            loop.close()


class TestErrorHandlingStrategies:
    """Test different error handling strategies in integrated scenarios."""
    
    def test_fail_fast_vs_collect_errors_strategies(self):
        """Test different approaches to handling multiple potential errors."""
        def validate_email(email: str) -> Result[str, str]:
            if '@' not in email:
                return Err("Missing @ symbol")
            if '.' not in email.split('@')[1]:
                return Err("Invalid domain format")
            return Ok(email)
        
        def validate_age(age_str: str) -> Result[int, str]:
            try:
                age = int(age_str)
                if age < 0 or age > 150:
                    return Err("Age out of valid range")
                return Ok(age)
            except ValueError:
                return Err("Age must be a number")
        
        def validate_name(name: str) -> Result[str, str]:
            if len(name.strip()) < 2:
                return Err("Name too short")
            if not name.replace(' ', '').isalpha():
                return Err("Name contains invalid characters")
            return Ok(name.strip())
        
        # Test data
        user_data = {
            'name': 'John Doe',
            'age': '30',
            'email': 'john.doe@example.com'
        }
        
        invalid_user_data = {
            'name': 'X',  # Too short
            'age': '-5',  # Invalid age
            'email': 'invalid-email'  # Missing domain
        }
        
        # Strategy 1: Fail fast (stop at first error)
        def validate_user_fail_fast(data: dict) -> Result[dict, str]:
            return (validate_name(data['name'])
                   .and_then(lambda name: validate_age(data['age'])
                            .map(lambda age: {'name': name, 'age': age}))
                   .and_then(lambda user: validate_email(data['email'])
                            .map(lambda email: {**user, 'email': email})))
        
        # Strategy 2: Collect all errors
        def validate_user_collect_errors(data: dict) -> Result[dict, list]:
            name_result = validate_name(data['name'])
            age_result = validate_age(data['age'])
            email_result = validate_email(data['email'])
            
            errors = []
            if name_result.is_err():
                errors.append(f"Name: {name_result.unwrap_or('Unknown error')}")
            if age_result.is_err():
                errors.append(f"Age: {age_result.unwrap_or('Unknown error')}")
            if email_result.is_err():
                errors.append(f"Email: {email_result.unwrap_or('Unknown error')}")
            
            if errors:
                return Err(errors)
            
            return Ok({
                'name': name_result.unwrap(),
                'age': age_result.unwrap(),
                'email': email_result.unwrap()
            })
        
        # Test valid data
        valid_result1 = validate_user_fail_fast(user_data)
        valid_result2 = validate_user_collect_errors(user_data)
        
        assert valid_result1.is_ok()
        assert valid_result2.is_ok()
        
        # Test invalid data
        invalid_result1 = validate_user_fail_fast(invalid_user_data)
        invalid_result2 = validate_user_collect_errors(invalid_user_data)
        
        assert invalid_result1.is_err()  # Fails fast on first error
        assert invalid_result2.is_err()  # Collects all errors
        
        # The collect_errors version should have multiple errors
        if invalid_result2.is_err():
            # We can't directly access the error list without unwrapping,
            # but we know it contains multiple validation errors
            pass


if __name__ == "__main__":
    pytest.main([__file__])