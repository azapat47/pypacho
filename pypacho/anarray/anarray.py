class AnArray:
    def transpose(self):
        pass
    def add(self,other):
        pass
    def subtract(self,other):
        pass
    def multiply(self,other):
        pass
    def divide(self,other):
        pass
    def iadd(self,other):
        pass
    def isub(self,other):
        pass
    def dot(self,other):
        pass
    def mod(self,other):
        pass
    def negative(self):
        pass
    def __str__(self):
        pass
    def __add__(self,other):             # A + B
        return self.add(other)
    def __iadd__(self, other):          # A += B
        return self.iadd(other)
    def __sub__(self,other):                  # A - B
        return self.subtract(other)  
    def __isub__(self, other):            # A-= B
        return self.iadd(other)    
    def __mul__(self,other):             # A * B (element-wise multiplication)
        return self.multiply(other)
    def __rmul__(self,other):
        return self.multiply(other)
    def __matmul__(self,other):          # A @ B (dot product)
        return self.dot(other)
    def __truediv__(self,other):         # A / B (element-wise)
        return self.divide(other)
    def __mod__ (self,other):            # A % B (element-wise)
        return self.mod(other)
    def __neg__(self):                   # -A
        return self.negative()
    def __pos__(self):                   # +A
        return self