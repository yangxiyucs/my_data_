
  def get(self, request,page):
        # 转为int
        page = int(page)

        # 获取当前页的数据
        try:
            skus_page = paginator.page(page)
        except EmptyPage:
            # 如果没有这一页 就去第一页
            page = 1
            skus_page = paginator.page(page)

        # 获取页码列表  要写到 EmptyPage异常处理之后  page才是正确的数据
        if paginator.num_pages <= 5:
            page_list = paginator.page_range
        elif page <= 3:
            page_list = range(1, 6)
        elif paginator.num_pages - page <= 2:
            page_list = range(paginator.num_pages - 4, paginator.num_pages + 1)
        else:
            page_list = range(page - 2, page + 3)

        context = {
            'sort': sort,
            'category': category,
            'categorys': categorys,
            'new_skus': new_skus,
            'skus_page': skus_page,
            'page_list': page_list,
        }

