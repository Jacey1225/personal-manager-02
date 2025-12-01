import ast
import multiprocessing as mp
import signal
import os
from typing import Set, List
import logging
import resource

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

class ValidateModules(ast.NodeVisitor):
    def __init__(self):
        self.found: List[str] = []

    def visit_Import(self, node: ast.Import):
        """Checks for regular imports in the code

        Args:
            node (ast.Import): The code logic tree to be validated
        """
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
    tree = ast.parse(code)
    validator = ValidateModules()
    validator.visit(tree)
    return validator.found

class ValidateRunTime:
    def __init__(self, timeout: float = 5.0, memory_limit: float = 1024.0):
        self.timeout = timeout
        self.memory_limit = memory_limit

    def run(self, 
            code: str,
            context: dict):
        """Tuns the code logic alongside a timer to cut off any operations that take a long time, eg. computationally exopensive operation with deep nested loops or infinite loops

        Args:
            code (str): The code logic to be validated for runtime
            context (dict): The context in which the code will be executed
        """
        def target(queue: mp.Queue):
            resource.setrlimit(
                resource.RLIMIT_AS, 
                (int(self.memory_limit), int(self.memory_limit)))
            
            try:
                local = {"result": None}
                exec(code, {"__builtins__": {}}, {**context, **local})
                queue.put(local["result"])
            except Exception as e:
                queue.put({"error": str(e)})
            except:
                queue.put({"error": "Unknown error occurred"})

        queue = mp.Queue()
        process = mp.Process(target=target, args=(queue,))
        process.start()
        process.join(self.timeout)

        if process.is_alive() and process.pid:
            os.kill(process.pid, signal.SIGTERM)
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            except:
                process.terminate()
            queue.put({"error": "Process timed out"})

        if not queue.empty():
            return {"result": queue.get()}
        return {"error": "Unknown error occurred"}