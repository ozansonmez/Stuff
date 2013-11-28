################################Newton-Raphson########################################
NR1=function(f,df, initial.value, iteration, tolerance) {
  
  x0 <- initial.value
  it<-iteration
  tol<-tolerance
  
  x=rep(0,it)
  x[1]=x0
  
  for (i in 2:it) 
  {
    
    x[i]= x[i-1]-f(x[i-1])/df(x[i-1])
    
    
    if(abs(x[i]-x[i-1])<tol){
      t.it=i
      break
    }
    t.it=it
    
  }
  
  out=cbind(1:t.it,x)
  dimnames(out)[[2]]=c("iteration","estimate")
  return(out[1:t.it,])
}
#Linkage example
g<-function(x){125/(2+x)-38/(1-x)+34/x}
dg<-function(x){-125/((1+x)^2)-38/((1-x)^2)-34/(x^2)}
NR1(g,dg,0.9,100,0.0000001)
######################################################################################