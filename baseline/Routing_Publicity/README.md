# Routing_Publicity

服务器获得明文 expert ID，直接将对应的明文权重应用于加密 activation。
实现位于 `public_routing.cpp`。

这是路由公开时的性能参照，会泄露 expert ID，不满足私有路由要求。
其密码学参数与其他 CKKS 对照一致。
