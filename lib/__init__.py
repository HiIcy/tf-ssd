# issue
"""
今天使用tensorflow，显存总是超，11G啊，说没就没了，可是我使用的变量全算下来，也就才消耗50M的显存。。。
google后，终于发现原因tf.gather(…)函数。
这个函数会初始化数组，保存取得的下标。下标用tf.IndexedSlices类保存，会在gather函数内隐式转换为大Tensor。而如果参数是Tensor，其中至少一维是’?’的话，那么恭喜你，中标了！
解决办法：把参数换成Variable，至少不要存在某个维的大小是’?’
"""
# 出现nan
# 看哪里计算出现了0或inf