from typing import Callable

class ValidateDiscussions:
    @staticmethod
    def validate_discussion(func: Callable):
        def wrapper(self, *args, **kwargs):
            discussion_id = kwargs.get("discussion_id")
            if not discussion_id:
                discussion_id = args[0]
                if not discussion_id:
                    raise ValueError(f"{func.__name__}, {func.__class__}: Discussion ID must be provided.")

            print(f"Validating discussion: {discussion_id}")
            result = func(self, *args, **kwargs)
            print(f"Validation result for discussion {discussion_id} from {func.__name__}: {result}")
            return result
        return wrapper
    
    @staticmethod
    def validate_new_discussion(func: Callable):
        def wrapper(self, *args, **kwargs):
            if not self.user_id or not self.project_id:
                raise ValueError(f"{func.__name__}, {func.__class__}: User ID and Project ID must be set.")

            for arg in args:
                if not arg:
                    raise ValueError(f"{func.__name__}, {func.__class__}: Invalid argument provided.")

            args_list = list(args)
            if not args_list[1]:
                args_list[1] = [self.user_id]

            if not args_list[2]:
                args_list[2] = []

            result = func(self, *tuple(args_list), **kwargs)
            print(f"Validation result for new discussion from {func.__name__}: {result}")
            return result
        return wrapper