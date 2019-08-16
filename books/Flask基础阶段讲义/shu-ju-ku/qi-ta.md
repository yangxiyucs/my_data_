# 信号机制

## Flask信号机制

- Flask信号(signals, or event hooking)允许特定的发送端通知订阅者发生了什么（既然知道发生了什么，那我们可以根据自己业务需求实现自己的逻辑）。
- Flask提供了一些信号（核心信号）且其它的扩展提供更多的信号。
- 信号依赖于**Blinker**库。
```
pip install blinker
```
- flask内置信号列表：http://docs.jinkan.org/docs/flask/api.html#id17
```python
template_rendered = _signals.signal('template-rendered')
request_started = _signals.signal('request-started')
request_finished = _signals.signal('request-finished')
request_tearing_down = _signals.signal('request-tearing-down')
got_request_exception = _signals.signal('got-request-exception')
appcontext_tearing_down = _signals.signal('appcontext-tearing-down')
appcontext_pushed = _signals.signal('appcontext-pushed')
appcontext_popped = _signals.signal('appcontext-popped')
message_flashed = _signals.signal('message-flashed')
```

### 信号应用场景

Flask-User 这个扩展中定义了名为 user_logged_in 的信号，当用户成功登入之后，这个信号会被发送。我们可以订阅该信号去追踪登录次数和登录IP：

```python
from flask import request
from flask_user.signals import user_logged_in

@user_logged_in.connect_via(app)
def track_logins(sender, user, **extra):
    user.login_count += 1
    user.last_login_ip = request.remote_addr
    db.session.add(user)
    db.session.commit()
```


## Flask-SQLAlchemy 信号支持

在 Flask-SQLAlchemy 模块中，0.10 版本开始支持信号，可以连接到信号来获取到底发生什么了的通知。存在于下面两个信号：

- models_committed
    - 这个信号在修改的模型提交到数据库时发出。发送者是发送修改的应用，模型 和 操作描述符 以 (model, operation) 形式作为元组，这样的元组列表传递给接受者的 changes 参数。
    - 该模型是发送到数据库的模型实例，当一个模型已经插入，操作是 'insert' ，而已删除是 'delete' ，如果更新了任何列，会是 'update' 。
- before_models_committed
    - 除了刚好在提交发送前发生，与 models_committed 完全相同。
    
```python
from flask_sqlalchemy import models_committed

# 给 models_committed 信号添加一个订阅者，即为当前 app
@models_committed.connect_via(app)
def models_committed(a, changes):
    print(a, changes)
```

> 对数据库进行增删改进行测试



