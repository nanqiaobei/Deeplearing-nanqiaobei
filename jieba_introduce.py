import jieba
seg_list=jieba.cut("我来到清华大学",cut_all=True)
print("/".join(seg_list))
#全部分词
seg_list1=jieba.cut("我来到清华大学了",cut_all=False)
print("/".join(seg_list1))
#精确分词
seg_list2=jieba.cut_for_search("小米硕士毕业于清华大学，博士毕业于斯坦福大学",HMM=False)
print("/".join(seg_list2))