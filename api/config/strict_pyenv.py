import ast
import multiprocessing as mp
import signal
import os
import sys
from typing import Set, List
import logging
import resource
import textwrap

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

"""Strict Python Environment Rules:
1. No access to file reading or writing using libraries such as 
os, sys, or any other standard library that interacts with the file system.
2. Restrict access to infinite loops, or computationally expensive
operations.
3. Limit access to predefined libraries by explicitly whitelisting allowed libraries.
4. Disallow the use of `eval()` and `exec()` functions to prevent arbitrary code execution.
5. Limit nesting depth as well as recursion depth
"""
FORBIDDEN_MODULES: Set[str] = {
    "os", "sys", "subprocess", "time", "signal", "shutil", "pathlib",
    "pickle", "nmap", "socket", "requests", "urllib", "threading",
    "multiprocessing", "asyncio", "concurrent", "builtins", "gc",
    "faulthandler", "traceback", "pty", "pwd", "grp", "resource", "termios",
    "boto3", "botocore", "google.cloud", "azure", "ctypes", "glob", "exec", "eval"
} 

safe_builtins = {
    'len': len, 'range': range, 'str': str, 'int': int, 'float': float,
    'bool': bool, 'list': list, 'dict': dict, 'tuple': tuple, 'set': set,
    'abs': abs, 'min': min, 'max': max, 'sum': sum, 'sorted': sorted,
    'enumerate': enumerate, 'zip': zip, 'map': map, 'filter': filter,
    'print': print, 'isinstance': isinstance, 'type': type,
}

class ValidateModules(ast.NodeVisitor):
    def __init__(self):
        self.found: List[str] = []

    def visit_Import(self, node: ast.Import):
        """Checks for regular imports in the code

        Args:
            node (ast.Import): The code logic tree to be validated
        """
        logger.info(f"Visiting import node: {ast.dump(node)}")
        for alias in node.names:
            name = alias.name.split(".")[0]
            if name in FORBIDDEN_MODULES:
                logger.warning(f"Forbidden module '{name}' imported.")
                self.found.append(name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        """Checks for import from a module in the code

        Args:
            node (ast.ImportFrom): The code logic tree to be validated
        """
        logger.info(f"Visiting import-from node: {ast.dump(node)}")
        module = node.module.split(".")[0] if node.module else ""
        if module in FORBIDDEN_MODULES:
            logger.warning(f"Forbidden module '{module}' imported.")
            self.found.append(module)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        """Checks for any functions calls that might be importing things in the code such as __import__()

        Args:
            node (ast.Call): The code logic tree to be validated
        """
        logger.info(f"Visiting call node: {ast.dump(node)}")
        func = node.func
        if isinstance(func, ast.Name):
            func_name = func.id
            if func_name in FORBIDDEN_MODULES:
                logger.warning(f"Forbidden function '{func_name}' called.")
                self.found.append(func_name)
            if node.args and isinstance(node.args[0], ast.Constant):
                if node.args[0].value in FORBIDDEN_MODULES:
                    logger.warning(f"Forbidden constant '{node.args[0].value}' used in function '{func.id}'.")
                    self.found.append(node.args[0].value)

        self.generic_visit(node)

def validate_code(code: str) -> List[str]:
    logger.info(f"Starting code validation...")
    code = textwrap.dedent(code)
    
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        logger.error(f"Syntax error in code: {e}")
        return ["syntax_error"]
    
    validator = ValidateModules()
    validator.visit(tree)
    if validator.found:
        logger.error(f"Validation failed. Forbidden modules found: {validator.found}")
    else:
        logger.info("Validation successful. No forbidden modules found.")
    return validator.found

def execute_code( 
                 context: dict, 
                 code: str, 
                 queue: mp.Queue):
        """Executes the given code provided by the given widget

        Args:
            context (dict): The context in which the code will be executed
            code (str): The code to be executed
            queue (mp.Queue): The multiprocessing queue to store the result or error

        Returns:
            mp.Queue: The multiprocessing queue containing the result or error
        """        
        try:

            local_scope = {**context}
            exec(code, {"__builtins__": safe_builtins}, local_scope)
        except Exception as e:
            queue.put({"error in execution": str(e)})

        result = None
        for key, value in local_scope.items():
            if callable(value) and not key.startswith("__") and key not in context:
                try:
                    result = value(context)
                except Exception as e:
                    queue.put({"error in function call": str(e)})
        if result is None:
            queue.put({"error in result": "No callable function found"})
        else:     
            queue.put(result)
class ValidateRunTime:
    def __init__(self, timeout: float = 5.0):
        self.timeout = timeout

    def run(self, 
            code: str,
            context: dict):
        """Tuns the code logic alongside a timer to cut off any operations that take a long time, eg. computationally exopensive operation with deep nested loops or infinite loops

        Args:
            code (str): The code logic to be validated for runtime
            context (dict): The context in which the code will be executed
        """
        logger.info(f"Raw code to run: {code}")
        new_code = textwrap.dedent(code)
        logger.info(f"Running processed function: {new_code}")

        try:
            queue = mp.Queue()
            process = mp.Process(target=execute_code, args=(context, new_code, queue))
            logger.info(f"Starting code execution with timeout {self.timeout}s...")
            process.start()
            logger.info("Joining process...")
            process.join(self.timeout)
        except Exception as e:
            logger.error(f"Error starting process: {str(e)}")
            return {"error": str(e)}

        if process.is_alive() and process.pid:
            logger.warning("Process timed out. Terminating...")
            os.kill(process.pid, signal.SIGTERM)
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            except:
                process.terminate()
            queue.put({"error": "Process timed out"})
        
        logger.info("Retrieving result from queue...")
        if not queue.empty():
            return {"result": queue.get()}
        return {"error": "Unknown error occurred, no result in queue"}