
# class Hello():
# 	def say_hello(self, name="world"):
# 		print("Hello,%s"%name)

def fn(self, name = "World"):
	print("Hello, %s"%name)

Hello = type("Hello",(object,),dict(say_hello=fn))
hello = Hello()
hello.say_hello()