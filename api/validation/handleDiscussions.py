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