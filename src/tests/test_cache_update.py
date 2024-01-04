from kifit.cache_update import update_fct
from kifit.cache_update import cached_fct
from kifit.cache_update import cached_fct_property


# class to test custom cache and update calls

class ClassName:

    def __init__(self):
        self._x = 0

    @cached_fct_property
    def get_n_cache_calls(self):
        if hasattr(self, 'n_cache_calls'):
            return self.n_cache_calls

    @update_fct
    def update(self, x):
        self._x = x

    @cached_fct
    def f(self, y):
        return 4 * self._x * y

    @cached_fct_property
    def h(self):
        return 4 * self._x


def test_cache_update():
    # create 2 instances to test that they dont interfere
    cn = ClassName()
    cn2 = ClassName()

    # initialise value
    cn.update(3)

    nloopcalls = 5

    assert cn.h == 12

    for i in range(nloopcalls):
        assert cn.h == 12, cn.h
        assert cn.f(17) == 204, cn.f(17)

    assert (cn.get_n_cache_calls[('h', ())] == nloopcalls), cn.get_n_cache_calls[('h', ())]
    assert (cn.get_n_cache_calls[('f', (17,))] == (nloopcalls - 1)), cn.get_n_cache_calls[('f', (17,))]

    cn2.update(4)

    nloopcalls2 = 6

    for i in range(nloopcalls2):
        assert (cn2.h == 16), cn2.h
        assert (cn2.f(37) == 592), cn2.f(37)

    assert (cn2.get_n_cache_calls[('h', ())] == (nloopcalls2 - 1)), cn2.get_n_cache_calls[('h', ())]
    assert (cn2.get_n_cache_calls[('f', (37,))] == (nloopcalls2 - 1)), cn2.get_n_cache_calls[('f', (37,))]

    cn.update(6)
    cn2.update(10)

    nloopcalls3 = 5

    for i in range(nloopcalls3):
        assert cn.h == 24, cn.h
        assert cn2.h == 40, cn2.h

    assert cn.f(17) == 408, cn.f(17)

    assert (cn.get_n_cache_calls[('f', (17,))] == 0), cn.get_n_cache_calls[('f', (17,))]
    assert (cn2.get_n_cache_calls[('h', ())] == (nloopcalls3 - 1)), cn2.get_n_cache_calls[('h', ())]


if __name__ == "__main__":
    test_cache_update()
