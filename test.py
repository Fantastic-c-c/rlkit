from rlkit.envs.wrappers import Wrapper

class A(object):
    def __init__(self):
        self.bar = 1

    def foo(self):
        print(self.bar)

class B(A):
    def __init__(self):
        super(B).__init__()
        self.bar = 2

class C(object):
    def __init__(self, inner):
        self.inner = inner
        self.bar = 3

    def foo(self):
        print('here')

    def __getattr__(self, attr):
        return getattr(self.inner, attr)

class D(Wrapper, object):
    def __init__(self, inner):
        super(D, self).__init__(inner)
        self.inner = inner
        self.bar = 4

b = B()
print(b.foo())

a = A()
c = C(a)
print(c.foo())

d = D(a)
print(d.foo())

