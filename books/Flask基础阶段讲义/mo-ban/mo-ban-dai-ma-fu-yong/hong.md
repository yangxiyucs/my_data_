# 宏

对宏(macro)的理解：
- 把它看作 Jinja2 中的一个函数，它会返回一个模板或者 HTML 字符串
- 为了避免反复地编写同样的模板代码，出现代码冗余，可以把他们写成函数以进行重用
- 需要在多处重复使用的模板代码片段可以写入单独的文件，再包含在所有模板中，以避免重复

## 使用

- 定义宏

```python
{% macro input(name,value='',type='text') %}
	<input type="{{type}}" name="{{name}}"
		value="{{value}}" class="form-control">
{% endmacro %}	
```
- 调用宏

```python
{{ input('name' value='zs')}}
```

- 这会输出

```python
<input type="text" name="name"
	value="zs" class="form-control">
```

- 把宏单独抽取出来，封装成html文件，其它模板中导入使用，文件名可以自定义macro.html

```python
{% macro function(type='text', name='', value='') %}
<input type="{{type}}" name="{{name}}"
value="{{value}}" class="form-control">

{% endmacro %}
```
- 在其它模板文件中先导入，再调用

```python
{% import 'macro.html' as func %}
{% func.function() %}	
```

## 代码演练

- 使用宏之前代码

```html
<form>
    <label>用户名：</label><input type="text" name="username"><br/>
    <label>身份证号：</label><input type="text" name="idcard"><br/>
    <label>密码：</label><input type="password" name="password"><br/>
    <label>确认密码：</label><input type="password" name="password2"><br/>
    <input type="submit" value="注册">
</form>
```

- 定义宏

```html
{#定义宏，相当于定义一个函数，在使用的时候直接调用该宏，传入不同的参数就可以了#}
{% macro input(label="", type="text", name="", value="") %}
<label>{{ label }}</label><input type="{{ type }}" name="{{ name }}" value="{{ value }}">
{% endmacro %}
```

- 使用宏

```html
<form>
    {{ input("用户名：", name="username") }}<br/>
    {{ input("身份证号：", name="idcard") }}<br/>
    {{ input("密码：", type="password", name="password") }}<br/>
    {{ input("确认密码：", type="password", name="password2") }}<br/>
    {{ input(type="submit", value="注册") }}
</form>
```




