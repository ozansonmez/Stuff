#HW4 #1

#Write an R/Python/C function for sampling truncated normal random variables 
#(possibly using a different algorithm). You may also use the code provide in the GitHub repo. 
#Sample 10,000 random variables from this function and verify the mean (roughly) matches the theoretical values.

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
SummaryTime[i,] = as.vector(time.cpu[1:3])
}

#Compare with theoretical values
a=dnorm((lower-mean))
b=dnorm((upper-mean))
A=pnorm((lower-mean))
B=pnorm((upper-mean))

#two tail truncation
E.X.2<-mean +(a-b)/(B-A)
S.X.2<-sqrt(1+((lower-mean)*a-(upper-mean)*b)/(B-A)-(E.X-mean)^2)

#upper tail truncation
E.X.U<-mean - b/B
S.X.U<-sqrt(1-(upper-mean)*(b/B) - (b/B)^2)

#lower tail truncation
E.X.L<-mean + (a/(1-A))
S.X.L<-sqrt(1-(a/(1-A)*((a/(1-A)-(lower-mean)))))

##plots
time.R.plot<-SummaryTime[,3]
#pulled values from RCUDA
time.RCUDA<-c(0.065,0.065,0.066,0.07,0.098,0.385,3.24,32.255)
png("PlotTime.png")
plot(1:8,time.R.plot,main="Plot of Total Computation Time in R and RCUDA,k=1-7",xlab="k, N=10^k",ylab="Total Time",ylim=c(0,6),type="l",col="blue")
lines(time.RCUDA,col='green',type='l',lwd=2)
legend(1,3,c("R","RCUDA"),lty=c(1,1),lwd=c(2.5,2.5),col=c("blue","Green"))
dev.off()
png("PlotTime1.png")
plot(1:8,time.R.plot,main="Plot of Total Computation Time in R and RCUDA,k=1-8",xlab="k, N=10^k",ylab="Total Time",ylim=c(0,35),type="l",col="blue")
lines(time.RCUDA,col='green',type='l',lwd=2)
legend(1,25,c("R","RCUDA"),lty=c(1,1),lwd=c(2.5,2.5),col=c("blue","Green"))
dev.off()
