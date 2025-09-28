"""LogManager class that provides logger instances with shared file and stream handlers"""
import os
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import datetime
import glob
from typing import Dict, List, Optional, Union

# Fixed LOG_DIR - using relative path to project logs directory
LOG_DIR = Path("./logs")

class SmartRotatingFileHandler(RotatingFileHandler):
    """
    A RotatingFileHandler that reuses existing log files during rotation
    if they haven't reached the size limit yet, preventing fragmentation.
    """
    
    def doRollover(self):
        """
        Do a rollover, but first check if there's an existing backup file
        that can be reused instead of creating a new one.
        """
        if self.stream:
            self.stream.close()
            self.stream = None
        
        # Check for existing backup files that can be reused
        existing_file = self._find_reusable_backup()
        
        if existing_file:
            # Use the existing file instead of creating new rotations
            self.baseFilename = existing_file
            if not self.delay:
                self.stream = self._open()
            return
        
        # If no reusable file found, proceed with normal rotation
        super().doRollover()
    
    def _find_reusable_backup(self) -> Optional[str]:
        """
        Find an existing active log file that can be reused.
        Only looks for .log files with the exact same base name.
        
        Returns:
            Path to reusable log file if found, None otherwise
        """
        base_dir = Path(self.baseFilename).parent
        base_name = Path(self.baseFilename).stem  # e.g., 'app' from 'app.log'
        
        # Get all .log files in the directory
        all_files = glob.glob(str(base_dir / "*.log"))
        
        # Filter to only files with the exact same base name
        log_files = []
        for file_path in all_files:
            file_base_name = Path(file_path).stem
            if file_base_name == base_name and file_path != self.baseFilename:
                log_files.append(file_path)
        
        # Check each file to see if it can be reused
        for file_path in log_files:
            try:
                if os.path.exists(file_path):
                    file_size = os.path.getsize(file_path)
                    # If file is under the size limit, we can reuse it
                    if file_size < self.maxBytes:
                        return file_path
            except (OSError, IOError):
                # Skip files we can't access
                continue
        
        return None

class LogManager:
    """
    A class that provides logger instances configured with file and stream handlers
    
    Attributes:
        log_dirs (List[Path]): List of directories where log files will be stored
        log_files (Dict[str, str]): Map of log file paths by directory
        log_file_size (int): Maximum size of log file before rotation in bytes
        backup_count (int): Number of backup log files to keep
        registered_loggers (dict): Dictionary to track registered loggers
        handlers (Dict[str, SmartRotatingFileHandler]): File handlers for each log file
    """

    # Standard logging levels for reference
    CRITICAL = 50
    ERROR = 40
    WARNING = 30
    INFO = 20
    DEBUG = 10
    NOTSET = 0
    
    def __init__(self, 
                 log_dir: Union[str, Path, List[Union[str, Path]]] = None, 
                 log_file_name: Union[str, Dict[str, str]] = None, 
                 log_file_size: int = 2097152, 
                 backup_count: int = 10,
                 debug_to_primary_only: bool = False):
        """
        Initialize the LogManager with configurable log directories and file names
        
        Args:
            log_dir: Directory or list of directories where log files will be stored.
                    Must be provided or an error will be raised.
            log_file_name: Name of the log file or dict mapping directory to file name.
                          If None, defaults to 'app.log'
            log_file_size: Maximum size of log file before rotation in bytes.
                          Defaults to 2MB
            backup_count: Number of backup log files to keep.
                         Defaults to 10
            debug_to_primary_only: If True, DEBUG logs will only go to the primary log file.
                                 INFO and above will go to all log files. Otherwise, all logs will go to all log files.
            
        Raises:
            ValueError: If log_dir is not provided
        """
        # Require log_dir parameter
        if log_dir is None:
            raise ValueError("log_dir parameter is required. You must specify a valid log directory.")
        
        # Store debug configuration
        self.debug_to_primary_only = debug_to_primary_only
        
        # Convert log_dir to a list of Path objects
        self.log_dirs = []
        if isinstance(log_dir, (str, Path)):
            self.log_dirs = [Path(log_dir)]
        else:
            self.log_dirs = [Path(d) for d in log_dir]
        
        # Create log directories if they don't exist
        for directory in self.log_dirs:
            try:
                if not os.path.exists(directory):
                    os.makedirs(directory, exist_ok=True)
            except OSError:
                print(f"Error: Creating directory. {directory}")
        
        # Set up log file names
        self.log_files = {}
        
        # Handle different log_file_name formats
        if log_file_name is None:
            # Default: Use 'app.log' for all directories
            for directory in self.log_dirs:
                self.log_files[str(directory)] = str(directory / "app.log")
        elif isinstance(log_file_name, str):
            # Single name: Use the same file name for all directories
            for directory in self.log_dirs:
                self.log_files[str(directory)] = str(directory / log_file_name)
        elif isinstance(log_file_name, dict):
            # Dict of names: Map directories to file names
            for directory in self.log_dirs:
                dir_key = str(directory)
                # Use the specified file name or default to 'app.log'
                file_name = log_file_name.get(dir_key, "app.log")
                self.log_files[dir_key] = str(directory / file_name)
                
        self.primary_log_file = self.log_files[str(self.log_dirs[0])]
        self.log_file_size = log_file_size
        self.backup_count = backup_count
        self.registered_loggers = {}
        
        # Track handlers for each log file
        self.handlers = {}
        
        # Configure formatters with UTC timestamps
        self.formatter = logging.Formatter(
            fmt="%(asctime)s UTC %(levelname)5.5s %(name)20.20s %(lineno)5d - %(message)s"
        )
        # Set UTC time for timestamps
        self.formatter.converter = lambda *args: datetime.datetime.now(datetime.timezone.utc).timetuple()
        
        # Create handlers for each log file using SmartRotatingFileHandler
        for dir_path, file_path in self.log_files.items():
            handler = SmartRotatingFileHandler(
                filename=file_path, 
                maxBytes=self.log_file_size, 
                backupCount=self.backup_count
            )
            
            # Primary handler always gets DEBUG level
            is_primary = dir_path == str(self.log_dirs[0])
            
            # Set appropriate level based on primary/non-primary status
            if is_primary or not self.debug_to_primary_only:
                handler.setLevel(logging.DEBUG)
            else:
                # For non-primary handlers with debug_to_primary_only=True, set to INFO
                handler.setLevel(logging.INFO)
                
            handler.setFormatter(self.formatter)
            self.handlers[dir_path] = handler
        
        # Primary handler is the first one (for backward compatibility)
        self.handler = self.handlers[str(self.log_dirs[0])]
        
        # Configure console output handler with UTC timestamps
        self.stream_formatter = logging.Formatter(
            fmt="%(asctime)s UTC %(levelname)5.5s %(name)20.20s - %(message)s"
        )
        # Set UTC time for stream formatter as well
        self.stream_formatter.converter = lambda *args: datetime.datetime.now(datetime.timezone.utc).timetuple()
        
        self.stream_handler = logging.StreamHandler()
        self.stream_handler.setLevel(logging.DEBUG)
        self.stream_handler.setFormatter(self.stream_formatter)

    def get_logger(self, name: str, log_dirs: List[str] = None, component_log_manager: 'LogManager' = None):
        """
        Provides a logger instance configured with file and stream handlers
        
        Args:
            name: The name of the logger
            log_dirs: Optional list of directory paths to log to
                     If None, logs to all directories
            component_log_manager: Optional component LogManager to incorporate
                                  If provided, its handlers will be added to this logger
            
        Returns:
            logging.Logger: Configured logger instance
        """
        logger = logging.getLogger(name)
        
        # Return existing logger if already registered and no component manager
        if name in self.registered_loggers and component_log_manager is None:
            return logger

        # Clear any existing handlers to avoid duplicates
        logger.handlers = []
        
        # Make sure propagate is False to avoid duplicate logs
        logger.propagate = False
        
        # Configure the logger with stream handler first
        logger.addHandler(self.stream_handler)
        
        # If component_log_manager is provided, integrate its handlers
        if component_log_manager is not None:
            # Always add the primary handler from this LogManager (for db_v1.log)
            primary_dir_key = str(self.log_dirs[0])
            logger.addHandler(self.handlers[primary_dir_key])
            
            # Add handlers from component_log_manager
            for dir_path, handler in component_log_manager.handlers.items():
                # Only add component handlers, not ours
                if dir_path not in self.handlers:
                    # If debug_to_primary_only is True, ensure component handlers get INFO+
                    if self.debug_to_primary_only:
                        handler.setLevel(logging.INFO)
                    logger.addHandler(handler)
                    
            # Ensure the debug_to_primary_only setting is respected
            if self.debug_to_primary_only:
                component_log_manager.set_debug_to_primary_only(True)
        else:
            # Configure with specified or all file handlers
            if log_dirs is None:
                # Add all handlers by default
                for handler in self.handlers.values():
                    logger.addHandler(handler)
            else:
                # Add only the specified handlers
                for dir_path in log_dirs:
                    dir_key = str(Path(dir_path))
                    if dir_key in self.handlers:
                        logger.addHandler(self.handlers[dir_key])
        
        # Always set logger level to DEBUG so it can pass messages to handlers
        logger.setLevel(logging.DEBUG)
        self.registered_loggers[name] = logger
        return logger

    def set_stream_level(self, level: int):
        """
        Sets the level of the stream handler
        
        Args:
            level: The logging level to set
            
        CRITICAL  50
        ERROR     40
        WARNING   30
        INFO      20
        DEBUG     10
        NOTSET    0
        """
        self.stream_handler.setLevel(level)
        
    def set_debug_to_primary_only(self, value: bool):
        """
        Configure whether DEBUG logs should go only to the primary log file
        
        Args:
            value: If True, DEBUG logs will only go to the primary log file.
                  If False, DEBUG logs will go to all log files.
        """
        if value == self.debug_to_primary_only:
            # No change needed
            return
            
        self.debug_to_primary_only = value
        
        # Update levels on all non-primary handlers
        for dir_path, handler in self.handlers.items():
            is_primary = dir_path == str(self.log_dirs[0])
            
            if not is_primary:
                if self.debug_to_primary_only:
                    # For non-primary handlers, only allow INFO and above
                    handler.setLevel(logging.INFO)
                else:
                    # Allow all levels
                    handler.setLevel(logging.DEBUG)

    def change_log_file(self, log_file_name: str, directory_index: int = 0):
        """
        Changes the log file used by a specific file handler
        
        Args:
            log_file_name: The new log file name
            directory_index: Index of the directory to change the log file for
                            Defaults to 0 (primary directory)
        """
        if directory_index < 0 or directory_index >= len(self.log_dirs):
            raise ValueError(f"Invalid directory index: {directory_index}")
            
        directory = self.log_dirs[directory_index]
        dir_key = str(directory)
        
        # Create a new file path
        new_file_path = str(directory / log_file_name)
        self.log_files[dir_key] = new_file_path
        
        if directory_index == 0:
            self.primary_log_file = new_file_path
        
        # Create a new file handler with the same settings but new file
        new_file_handler = SmartRotatingFileHandler(
            filename=new_file_path,
            maxBytes=self.log_file_size,
            backupCount=self.backup_count,
        )
        
        # Set appropriate level based on primary/non-primary status
        is_primary = dir_key == str(self.log_dirs[0])
        if is_primary or not self.debug_to_primary_only:
            new_file_handler.setLevel(logging.DEBUG)
        else:
            # For non-primary handlers with debug_to_primary_only=True, set to INFO
            new_file_handler.setLevel(logging.INFO)
            
        new_file_handler.setFormatter(self.formatter)

        # Get the old handler
        old_handler = self.handlers[dir_key]
        
        # Update all registered loggers to use the new handler
        for logger in self.registered_loggers.values():
            # Check if the logger has the old handler
            if old_handler in logger.handlers:
                logger.removeHandler(old_handler)
                logger.addHandler(new_file_handler)
        
        # Close the old handler
        old_handler.close()
        
        # Update handler references
        self.handlers[dir_key] = new_file_handler
        if directory_index == 0:
            self.handler = new_file_handler

    def cleanup_oversized_duplicates(self):
        """
        Clean up duplicate log files that may have been created by multiple handlers.
        Keeps the most recent file and removes older duplicates.
        """
        for dir_path, file_path in self.log_files.items():
            self._cleanup_duplicates_for_file(file_path)
    
    def _cleanup_duplicates_for_file(self, base_log_path: str):
        """
        Clean up duplicate active log files for a specific log file.
        Only removes excess .log files with the exact same base name.
        
        Args:
            base_log_path: Path to the base log file
        """
        base_dir = Path(base_log_path).parent
        base_name = Path(base_log_path).stem  # e.g., 'app' from 'app.log'
        
        # Get all .log files in the directory
        all_files = glob.glob(str(base_dir / "*.log"))
        
        # Filter to only files with the exact same base name
        active_log_files = []
        for file_path in all_files:
            file_base_name = Path(file_path).stem
            if file_base_name == base_name and file_path != base_log_path:
                active_log_files.append(file_path)
        
        if len(active_log_files) <= self.backup_count:
            return
        
        # Sort by modification time (newest first)
        active_log_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        
        # Keep only the configured number of active log files
        files_to_keep = active_log_files[:self.backup_count]
        files_to_remove = active_log_files[self.backup_count:]
        
        # Remove excess active log files
        for file_path in files_to_remove:
            try:
                os.remove(file_path)
                print(f"Removed excess active log file: {Path(file_path).name}")
            except Exception as e:
                print(f"Warning: Could not remove active log file {file_path}: {e}") 