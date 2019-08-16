# 静态化首页的手动脚本

为了方便开发，随时生成静态化首页，我们可以在scripts中新建静态化首页的脚本

regenerate_index_html.py

```python
#!/usr/bin/env python

"""
功能：手动生成所有SKU的静态detail html文件
使用方法:
    ./regenerate_index_html.py
"""
import sys
sys.path.insert(0, '../')

import os
if not os.getenv('DJANGO_SETTINGS_MODULE'):
    os.environ['DJANGO_SETTINGS_MODULE'] = 'meiduo_mall.settings.dev'

 # 让django进行初始化设置
import django
django.setup()


from contents.crons import generate_static_index_html


if __name__ == '__main__':
    generate_static_index_html()
```

为文件增加可执行权限

```shell
chmod +x regenerate_index_html.py
```

