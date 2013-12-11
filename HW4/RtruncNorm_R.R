#HW4 CPU R code

library(truncnorm)
truncnorm<-function(k,mean,sd,lower,upper){rtruncnorm(10^k,lower,upper,mean,sd)}
SummaryTime<-matrix(0,8,3)
Summary<-matrix(0,8,2)
mean = 0
sd = 1
lower = -Inf
upper = -10
for (i in 1:8){
time.cpu<-system.time({truncnorm.cpu = truncnorm(i,mean,sd,lower,upper)})
Summary[i,1]=mean(truncnorm.cpu)
Summary[i,2]=sd(truncnorm.cpu)
Result_Time[i,] = as.vector(time.cpu[1:3])
}


##plots
time.R.plot<-Result_Time[,3]
#pulled values from RCUDA
time.RCUDA<-c(0.056,0.069,0.057,0.061,0.089,0.375,3.24,32.3)
png("PlotTime.png")
plot(1:8,time.R.plot,main="Total Time comparison for R and RCUDA",xlab="k=1,...,8",ylab="Total Time",ylim=c(0,160),type="l",col="red")
lines(time.RCUDA,col='green',type='l',lwd=2)
legend(1,100,c("R","RCUDA"),lty=c(1,1),lwd=c(2.5,2.5),col=c("red","green"))
dev.off()

