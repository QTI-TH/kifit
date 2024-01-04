USE_CACHE = True

# use lru_cache ?


def cached_fct(old_func):
    """
    Decorate functions to cache return value.
    """
    def new_func(self, *args, **kwargs):
        if USE_CACHE:
            # set-up cache registry on class instance
            if not hasattr(self, 'function_cache'):
                self.function_cache = {}
                self.n_cache_calls = {}  # counts cache lookups

            # ASSUMING KWARGS=0, and ARGS simple integers
            cache_tag = (old_func.__name__, tuple(args))
            assert len(kwargs) == 0

            if (cache_tag not in self.function_cache) or (self.function_cache[cache_tag] is None):
                ret = old_func(self, *args, **kwargs)
                self.function_cache[cache_tag] = ret
                self.n_cache_calls[cache_tag] = 0
            else:
                self.n_cache_calls[cache_tag] += 1

            return self.function_cache[cache_tag]
        else:
            return old_func(self, *args, **kwargs)

    return new_func


def cached_fct_property(old_func):
    """
    Makes a property out of `cached_fct`.
    """
    @property
    def new_func(self):
        return cached_fct(old_func)(self)

    return new_func


def update_fct(old_func):
    """
    Decorates function to reset cache of cached functions.
    """
    def new_func(self, *args, **kwargs):
        if hasattr(self, 'function_cache'):
            cache = getattr(self, 'function_cache')
            for f in cache:
                cache[f] = None

        return old_func(self, *args, **kwargs)

    return new_func
