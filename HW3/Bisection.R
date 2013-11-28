#define the function parameters
#f=function, l=lower bound, u=upper bound, tol=tolerance
bisection <- function(f, l, u, tol,maxiteration) { 
f.l = f(l) 
f.u = f(u) 
d = abs(f(l)-f(u))
iteration=1

#giving a warning for wrong choice of lower and upper values of the interval
if (f.l * f.u > 0) stop("f(u)*f(l)>0, choose different interval!\n")

while(d > tol & iteration < maxiteration){
  c = (l+u)/2 
  f.c = f(c)
  if (abs(f.c) <= tol) break 
  if(f.c * f.u <0) {l<-c}
  if(f.c * f.l <0) {u<-c}
  d = abs(f(l)-f(u)) #find the new difference each time calculating new c
  iteration=iteration+1 #count the number of function evaluations
}
list(x = c, value = f.c, "number of iterations"=iteration) #list all the info
}

#################################for linkage ptoblem############################
g<-function(x){125/(2+x)-38/(1-x)+34/x}
bisection(g,0.01,0.99,0.0000001,100)