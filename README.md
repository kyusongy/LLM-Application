# LLM-Application

This repository contains a framework for building Large Language Model (LLM) applications with organized components and utilities.

## Project Structure

```
|── common/            # Common data type definitions and utilities
|
├── prompt/           # Prompt templates and configurations
│
├── llm/              # LLM client implementations
│
├── project/          # Project implementation scripts
│
├── utils/            # Utility functions and helpers
```

## Component Description

### Common (/common)
Contains core data type definitions and basic utilities that are used across the project:
such as:

- `llm.py`: Defines LLM-related data structures and types
- `tool.py`: Implements utility tools and helper functions

### Prompts (/prompt)
Organizes different types of prompts in separate folders:
- Each feature has its own directory with `system.yaml` and `user.yaml` configurations
- Includes implementations for argument filling and meta prompt handling

### LLM Client (/llm)
Maintains LLM client implementations:
such as:

- `sync_client.py`: Synchronous client for interacting with LLM services

### Project Scripts (/project)
Contains implementation scripts for specific functionalities:
such as:

- `error_fixer.py`: Implementation for automatic error fixing

### Utilities (/utils)
Common utilities and helper functions:
- `repair.py`: Utilities for json repair and maintenance

## Usage

## Contributing

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
