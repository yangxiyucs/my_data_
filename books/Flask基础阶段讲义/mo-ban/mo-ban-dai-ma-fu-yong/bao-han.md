# 包含

Jinja2模板中，除了宏和继承，还支持一种代码重用的功能，叫包含(Include)。它的功能是将另一个模板整个加载到当前模板中，并直接渲染。

- include的使用

```python
{% include 'hello.html' %}
```

包含在使用时，如果包含的模板文件不存在时，程序会抛出**TemplateNotFound**异常，可以加上 `ignore missing` 关键字。如果包含的模板文件不存在，会忽略这条include语句。

- include 的使用加上关键字ignore missing

```python
{% include 'hello.html' ignore missing %}
```


# 小结
- 宏(Macro)、继承(Block)、包含(include)均能实现代码的复用。
- 继承(Block)的本质是代码替换，一般用来实现多个页面中重复不变的区域。
- 宏(Macro)的功能类似函数，可以传入参数，需要定义、调用。
- 包含(include)是直接将目标模板文件整个渲染出来。