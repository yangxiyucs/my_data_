# coding=utf-8
class Student(object):
    """装饰器 property 属性设置"""

    def __init__(self, name):
        self.name = name
        self.name = None

    @property
    def age(self):
        return self._age

    @age.setter
    def age(self, value):
        if not isinstance(value, int):
            raise ValueError('输入不合法：年龄必须为数值!')
        if not 0 < value < 100:
            raise ValueError('输入不合法：年龄范围必须0-100')
        self._age = value

    @age.deleter
    def age(self):
        del self._age


xiaoming = Student("小明")

# 设置属性
xiaoming.age = 25

# 查询属性
print(xiaoming.age)

# 删除属性
del xiaoming.age


# -----------------------------------------------------------------------
def say_hello(contry):
    # 三层封装 装饰器传参
    def wrapper(func):
        def deco(*args, **kwargs):
            if contry == "china":
                print("你好!")
            elif contry == "america":
                print('hello.')
            else:
                return

            # 真正执行函数的地方
            func(*args, **kwargs)

        return deco

    return wrapper


# 小明，中国人
@say_hello("china")
def xiaoming():
    pass


# jack，美国人
@say_hello("america")
def jack():
    pass

# ----------------------------------------------------------------------
