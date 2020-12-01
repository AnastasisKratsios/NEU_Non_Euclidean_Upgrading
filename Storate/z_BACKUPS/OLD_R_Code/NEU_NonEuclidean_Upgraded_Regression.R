### DATA
Goog.dat<-as.numeric(sp.500$GOOG[!is.na(sp.500$GOOG)])
Appl.dat<-as.numeric(sp.500$AAPL)
### Synchronising timeseries
len<-min(length(Appl.dat),length(Goog.dat))
Appl.dat<-Appl.dat[c((length(Appl.dat)-len+1):length(Appl.dat))]
##


# Meta-meta-parameters
#-----------------------#
#rm("First.run")
gc()

# Parameters
#-------------------#
if(TRUE){
set.seed(123)
dorku<-10^3
x.granulatity<-10^3
y.noise<-.5 #sigma in paper's table

tol<-10^-2 # Iterations tolerance

# Extras
#----------#
# Well performing cap is 7!!!
locality.cap<-7#;rm(locality.cap) # Note: use different cap on first example when sigma is high
X.speed.fact<-10^-1 #Reduce X bump speed
manual.override<-c(.7,.3) #OVERRIDES HOW much randomness is selected
normalize.it<- FALSE#Normalize at every iteration
#
#
#
#
#




# INITIALIZE PLOTS
N.plots<-floor(sqrt(length(dorku)))
par(mfrow=c(N.plots,N.plots))
# INITIALIZE Stats
#------------------------#
# INI. estimated Squared errors
ESE<-c()
# INI. estimated errors
Error.estimates<-c()
# INITIALIZE HISTORY
X.hist<-c.hist<-s.hist<-c()
  
# INITIALIZE ITERATION
N.iterations<-dorku
###
# Define function to non-parametrically regress
#-------------------------------------------------#
f.name<-"I(x<.5)"
f<-function(x){
  #min(max(x^2/((x-2)*(x-6)),-1000),1000)#CUTTOFS ADDED TO PREVENT EXPLOSION ERRORS
  #if(x<.5){cos(exp(-x))*sqrt(abs(x))+1}else{exp(-cos(x))}
  #cos(exp(-x))*sqrt(abs(x))+1
  #cos(x^2)*exp(-x)
  #cos(exp(-x))
  #if(x<.5){1}else{0}
  #min(exp(tan(x)),x+cos(x))
  min(exp(-1/(x+1)^2),x+cos(x))
  #if(x<.5){1}else{0}
};f<-Vectorize(f)


#--------------------------------------------#
# Defining Things
#--------------------------------------------#
colfunc0 <- colorRampPalette(c("forestgreen","mediumorchid4"))
colfunc <- colorRampPalette(c("mediumorchid4", "midnightblue"))
colfunc2 <- colorRampPalette(c("midnightblue", "navyblue"))
colfunc3 <- colorRampPalette(c("navyblue", "cyan3"))
colvect<-c(colfunc0(round(N.iterations*.1,0)),colfunc(round(N.iterations*.4,0)),colfunc2(round(N.iterations/4,0)),colfunc3(round(N.iterations/4,0)))
# 
dat.x<-seq(from=-3,to=3,length.out = x.granulatity)
dat.y<-f(dat.x)+rnorm(length(dat.x))*y.noise
dat.x<-l.inf.normalizacion(dat.x)
dat.y<-l.inf.normalizacion(dat.y)
y.dataframe<-x.dataframe<-cbind(dat.x,dat.y)
plot(dat.x,dat.y,pch=20,col=alpha("seagreen1",.5),type="o",main=paste(dorku[i.dec],"Reconfigurations"),xlab="",ylab=""#,ylim = c(-.5,1.5)
     )

####################---####################-#---------------#
# Data-grabbing and Function initializing section
####################---####################-#---------------#
if(exists("First.run")){# FIRST.RUN<-FALSE
  First.run<-":)"
  #-############ -----------------------------------################-#
  # Load required packages if available...if not install - BEGIN
  #-############ -----------------------------------################-#
  # Check for missing packages
  list.of.packages <- c("ggplot2","knitr","PerformanceAnalytics","zoo","xts","tseries","MASS","mvtnorm","quantmod","glmnet","utils","beepr","bootstrap","boot","stats","graphics","splines","pspline","boot","xtable","KernSmooth","locfit","scales","neuralnet")
  new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
  # Download missing packages
  if(length(new.packages)) install.packages(new.packages)
  
  # Load packages requried for code...
  lapply(list.of.packages, require, character.only = TRUE)
  #-############ -----------------------------------################-#
  # Load required packages if available...if not install - END
  #-############ -----------------------------------################-#
  # FUNCTIONS
  #------------#
  
  # Confidence Intervals
  #---------------------------------#
  ## BOOTSTRAP and RELATED FUNCTIONS
  # AUXILIARY
  # function to obtain the mean
  Bmean <- function(data, indices) {
    d <- data[indices] # allows boot to select sample 
    return(mean(d))
  } 
  # BOOTSTRAPPING FUNCTION ITSELF
  btstrapped.confidence.interval.bca<-function(data.in,granularity.in=1000){
    granularity.in<-max((length(data.in)/2),1000,granularity.in)
    # bootstrapping with 1000 replications 
    boot.results <- boot(data=data.in, statistic=Bmean, R=granularity.in)
    # get 95% confidence interval 
    out.boot<-boot.ci(boot.results, type=c("norm", "basic", "perc", "bca")) #computes using methods
    out.boot<-as.numeric(out.boot$bca)[c(4,5)] # select the skewness and bias-corrected bootstrap method and unload the endpoints
    out.boot<-c(out.boot[1],Bmean(data.in),out.boot[2])
    names(out.boot)<-c("95.l","mean","95.u")
    return(out.boot)
  }
  
  
  # Reconfigurations
  #---------------------------------#
  
  # Lumps and bumps
  norm.e<-function(x){sqrt(sum(x^2))};norm.e2<-function(x){sum(x^2)}
  bump.f<-function(x,sigma.in){if(norm.e2(x)<sigma.in){(exp(-(sigma.in/(sigma.in-norm.e2(x)))^2))/exp(sigma.in)}else{0}}
  # Polynommial bump
  #bump.f<-function(x,sigma.in){if(norm.e2(x)<sigma.in){
  #  ((x-sigma.in)^2)*((x+sigma.in)^2)/sigma.in^4
  #}else{0}}
  RDR<-function(x,X.in,s.in,c.in){c(x[1],x[2]+X.in*(
    bump.f(abs((x[1]-c.in[1])),s.in)
  )
  )}

  
}
# INITIALIZE EXTERNAL PARAMETER(s)
update.me.thanks<-length(dat.x) # NUMBER OF x points
}






#
#
#
#
#
# Autoexecution failsafe... space :>
#
#
#
#
#
print("READY BLORNYUS") # FAILSAFE EXECUTES 1 instead of running code


# Write system time
NEU.OLS.start.time<-Sys.time()

# Run from here 
for(i in 1:length(colvect)){
# IMEDIATLY BREAKS IF INSIGNIFICANT ERROR  
if(max(abs(x.dataframe[,2]-.5))<tol){print("kakak_keke_party!!!WOOOO!");break}
if(max(abs(x.dataframe[,2]))>2){print("keke_party!!!WOOOO!");break}

    
# DETERMINES POINT OF FURTHEST DISTANCE AND PULLS IT IN
lines(x=x.dataframe[,1],y=rep(.5,update.me.thanks),col=alpha("green",0.3))

# Builds target
#--------------------#
# Removes residules which are out of the box defined by minimimal and maximal values of function on compact interval considered
if(sample(c(TRUE,FALSE),prob=manual.override,size = 1)){# CLASSICAL
  target<-which.max(abs(x.dataframe[,2]-.5))}else{#RANDOM
    target<-sample(c(1:update.me.thanks),size=1)
}
# TEST which are above or below
signs<-numeric(update.me.thanks)
signs1<-as.numeric(x.dataframe[,2])<.5
signs[signs1==TRUE]<- -1
signs[signs1==FALSE]<- 1;rm(signs1)
# CLASSIFIES TARGETS
#--------------------#
locality<-c(0,diff(signs));locality[locality==0]<-NA
# Categorize in terms of crossings
locality[!is.na(locality)]<-c(1:sum(!is.na(locality)))
locality<-rev(na.locf(rev(locality)))
locality<-locality[-length(locality)]# last one is misclassified so kill and fix in the next step along with the beautiful others :D
# Adds in any missing tail..since na.locf only interpolated backwards and forwards from last point of NA crossing...:/
vector.to.append<-rep((tail(locality,n=1)+1),(update.me.thanks-length(locality)))
locality<-append(locality,vector.to.append);rm(vector.to.append)
# Boolean Locality Vector
locality<-locality==locality[target]


########################################-#
# Determins Parameters
########################################-#
# Determins c
#-------------------#
c.int<-x.dataframe[target,]
points(c.int[1],c.int[2],col="orange",pch=20);c.int
# Determins sigma
#-------------------#
s.int<-max((1/update.me.thanks),(sum(locality)/update.me.thanks))/2 #failsafe to prevent 0 
# Determins X
#-------------------#
X.int<-as.numeric(abs(c.int-.5)[2])
X.int<- (-signs[target])*X.int*X.speed.fact


# FAILSAFE(s)
#-----------------#
# Enfore locality cap if exceds box
if((min(x.dataframe[,2])< (-.5) |max(x.dataframe[,2])> (1.5))){
  s.int<-  (1/update.me.thanks)
}

#-----------------------------------#
#-----------------------------------#
#### Wrwrite Reconfiguration
#-----------------------------------#
#-----------------------------------#
y.dataframe<-x.dataframe
# TEST ZONE _---____-------____---#
#bump.f(abs(x.dataframe[4,1]-c.int[1]),s.int)
#test<-as.numeric(RDR(x=c.int,X.in=X.int,s.in=s.int,c.in=c.int))
#points(c.int[1],c.int[2],pch=22)
#points(test[1],test[2],col="red")
# TEST ZONE _---____-------____---#

# INITIALIZATIONS
for(k in 1:update.me.thanks){
  y.update<-x.dataframe[k,]
  y.dataframe[k,]<-as.numeric(RDR(x=y.update,X.in=X.int,s.in=s.int,c.in=c.int))
}
points(x=y.dataframe[,1],y.dataframe[,2],col="red",pch=20)
# UPDATE
x.dataframe<-y.dataframe
  if(normalize.it==TRUE){
    x.dataframe[,1]<-l.inf.normalizacion(y.dataframe[,1])
    x.dataframe[,2]<-l.inf.normalizacion(y.dataframe[,2])
  }
alpha.current<- min(i/N.iterations,.5)
lines(x.dataframe[,1],x.dataframe[,2],pch=20,col=alpha(colvect[i],alpha.current))

# UPDATE HISTORY
X.hist<-append(X.hist,X.int)
s.hist<-append(s.hist,s.int)
c.hist<-rbind(c.hist,c.int)


# Prints worst residual from past update
print(paste("X:",X.int))#;print(paste("c:",c.int))#;print(paste("s:",s.int))
}
#

plot(x=x.dataframe[,1],y=x.dataframe[,2],pch=20,col=alpha("navyblue",.8))
#points(x=x.dataframe[,1],y=x.dataframe[,2],pch=20,col=alpha("red",.8))
fit.plott<-lm(as.numeric(x.dataframe[,2])~as.numeric(x.dataframe[,1]));fit.plot<-as.numeric(fit.plott$fitted.values)
lines(x=x.dataframe[,1],y=fit.plot,pch=20,col=alpha("cyan1",.8),lwd=2)
#legend("bottomleft", 95, legend=c("Rec. Dat.","Dat.","OLS"),       col=c("navyblue","seagreen1","cyan1"), pch=c(20,20), cex=0.4)
# STATISTICAL ESTIMATES
# Estimated Mean Squared Errors
ESE<-append(ESE,mean((fit.plott$residuals)^2))
# Estimated Mean Squared Errors
Error.estimates<-rbind(Error.estimates,btstrapped.confidence.interval.bca(fit.plott$residuals))

### FIT NEU-Local Regression
fit.plott.loc<-locfit(as.numeric(x.dataframe[,2])~lp(as.numeric(x.dataframe[,1]), nn=0, h=0.5, deg=1));fit.plot.loc<-as.numeric(fit.plott$fitted.values)
fit.plott.loc<-predict(fit.plott.loc,as.numeric(x.dataframe[,1]))
NEU.loc.res<-(as.numeric(x.dataframe[,1])-fit.plott.loc);NEU.loc.res<-NEU.loc.res^2

# Estimated Mean Squared Errors
ESE<-append(ESE,mean(NEU.loc.res^2))
# Estimated Mean Squared Errors
Error.estimates<-rbind(Error.estimates,btstrapped.confidence.interval.bca(NEU.loc.res))




#plot(x=as.numeric(x.dataframe[,1]),y=fit.plot,pch=20,col=alpha("cyan1",.8),lwd=2)
#plot(x=as.numeric(x.dataframe[,1]),y=as.numeric(x.dataframe[,2]),pch=20,col=alpha("cyan1",.8),lwd=2)
#--------------------------------------------------------------------#
#                           DECONFIGURATION
#--------------------------------------------------------------------#
# Required Function(s)
#-----------------------#
# Writes deconfiguration function
deconfig.vect<-function(y,X.in,s.in,c.in){c(y[1],y[2]-X.in*(
  bump.f(abs((y[1]-c.in[1])),s.in)
))
}


# Performs Deconfiguration
#---------------------------#
# Reverses Parameters in order to deconfigure
X.hist<-rev(X.hist)
s.hist<-rev(s.hist)
c.hist<-c.hist[rev(c(1:nrow(c.hist))),]
# Initializes deconfigured data
x.dec.dat<-cbind(x.dataframe[,1],fit.plot)
# Performs deconfiguration
for(i.dec in 1:length(X.hist)){
  X.hist.i<-X.hist[i.dec]
  s.hist.i<-s.hist[i.dec]
  c.hist.i<-c.hist[i.dec,]
# Picks point  
  for(j.dec in 1:nrow(x.dec.dat)){
   # Pics point
    killa<-x.dec.dat[j.dec,]
   # Applies deconfigration
    killa<-deconfig.vect(y=killa,X.in=X.hist.i,c.in=c.hist.i,s.in=s.hist.i)
   # Sends him home back into the dataframe
    x.dec.dat[j.dec,]<-killa
  }
}
#
# WRITESE SYSTEM TIME
NEU.OLS.end.time<-Sys.time()
# Writes Errors
EMSE.NEU.OLS<-mean((dat.y.killa-dat.y)^2)
mean.NEU.OLS.confid.interval<-btstrapped.confidence.interval.bca(dat.y.killa-dat.y)

# Deconfigure NEU-Loc-Reg
# Performs Deconfiguration
#---------------------------#
# Reverses Parameters in order to deconfigure
X.hist<-rev(X.hist)
s.hist<-rev(s.hist)
c.hist<-c.hist[rev(c(1:nrow(c.hist))),]
# Initializes deconfigured data
x.dec.dat.loc<-cbind(x.dataframe[,1],fit.plott.loc)
# Performs deconfiguration
for(i.dec in 1:length(X.hist)){
  X.hist.i<-X.hist[i.dec]
  s.hist.i<-s.hist[i.dec]
  c.hist.i<-c.hist[i.dec,]
  # Picks point  
  for(j.dec in 1:nrow(x.dec.dat.loc)){
    # Pics point
    killa.loc<-x.dec.dat.loc[j.dec,]
    # Applies deconfigration
    killa.loc<-deconfig.vect(y=killa.loc,X.in=X.hist.i,c.in=c.hist.i,s.in=s.hist.i)
    # Sends him home back into the dataframe
    x.dec.dat.loc[j.dec,]<-killa.loc
  }
}
# Stats
EMSE.NEU.loc<-mean(((dat.y.killa.loc-dat.y))^2)
mean.NEU.OLS.confid.interval.loc<-btstrapped.confidence.interval.bca(dat.y.killa.loc-dat.y)



# Benchmark Against local regression
#------------------------------------#
loc.start.run<-Sys.time()
# Performs Local Regression
loc.fitt<-locfit(dat.x ~ lp(dat.y, nn=0, h=0.5, deg=1))
loc.fit<-predict(loc.fitt,dat.y)
loc.res.na<-(dat.y-loc.fit);loc.res<-loc.res.na^2
# Estimated Mean Squared Errors
ESE.loc<-as.matrix(mean(loc.res))
# Estimated Mean Squared Errors
Error.estimates.loc<-btstrapped.confidence.interval.bca(loc.res.na)
loc.end.run<-Sys.time()

# Benchmark Against p-splines regression
#------------------------------------#
p.start.run<-Sys.time()
# Performs Local Regression
test<-sm.spline(x=dat.x, y=dat.y, df=5)
p.fit<-predict(sm.spline(x=dat.x, y=dat.y, df=5), dat.y)
p.res.na<-(dat.y-p.fit);p.res<-p.res.na^2
# Estimated Mean Squared Errors
ESE.p<-as.matrix(mean(p.res))
# Estimated Mean Squared Errors
Error.estimates.p<-btstrapped.confidence.interval.bca(p.res.na)
p.end.run<-Sys.time()


# Benchmark Against Feed-Forward Neural Network
#------------------------------------#--------------#
# Prepare Data for Neural-Net Package
#----------------------------------------#
# Data Generation
# Training Data
train.in<-dat.x#as.data.frame(seq(from=1,to=5,length.out = 50))
train.out<-dat.y#train.in^2
# Test Data
test.in<-dat.x#as.data.frame(seq(from=-10,to=10,length.out = 100))
test.out<-dat.y#test.in^2
# Formatting
trainingdata<-cbind(train.in,train.out)
testdata<-cbind(test.in,test.out)
colnames(trainingdata)<-c("x","fx")
colnames(testdata)<-c("x","fx")


# Build FFANN
ANN.start.run<-Sys.time()
learn.quick<-neuralnet(formula = fx~x,
                       data = trainingdata,hidden = c(300,100,200,50,100,100),
                       threshold = 0.01,
                       lifesign = "full",
                       lifesign.step = 10
)
pred.learn.quick<-predict(learn.quick,testdata)
lines(x=dat.x,y=pred.learn.quick,type="l",col="purple",lwd=1.5)
ANN.MSE<-btstrapped.confidence.interval.bca((pred.learn.quick-f(dat.x)))
EMSE.ANN<-mean(((pred.learn.quick-f(dat.x)))^2)
ANN.end.run<-Sys.time()







# FINAL FIT Visualization
#-----------------------------------------------#
par(mfrow=c(1,1))
# PLOT RESIDUALS
#------------#
# NEU Regression
# Writes data
dat.x.killa<-x.dec.dat[,1]#l.inf.normalizacion(x.dec.dat[,1])
dat.y.killa<-x.dec.dat[,2]#l.inf.normalizacion(x.dec.dat[,2]) # Wierd flip happens some times... so we flip it back...seems to be a coding kaka
# Writes Residuals
loc.res.na<-(dat.y-loc.fit)
NEU.res.na<-(dat.y-x.dec.dat[,2])
# NEU Local Regression 
dat.x.killa.loc<-x.dec.dat.loc[,1]#l.inf.normalizacion(x.dec.dat.loc[,1])
dat.y.killa.loc<-x.dec.dat.loc[,2]#l.inf.normalizacion(x.dec.dat.loc[,2]) # Wierd flip happens some times... so we flip it back...seems to be a coding kaka
NEU.res.na<-(dat.y-x.dec.dat.loc[,2])

# Plots
plot(dat.x,NEU.res.na,col=alpha("red",.5),main=paste("NEU-OLS Residuals"),xlab="",ylab="Residuals",pch=20)
#plot(x=dat.x.killa,y=rep(0,length(dat.x)),type="l",col="seagreen",lwd=3)
plot(x=dat.x,y=loc.res.na,col="blue",lty=2,pch=20,main=paste("LOESS Residuals"),xlab="",ylab="Residuals")
plot(x=dat.x,y=p.res.na,col="yellow",lty=2,pch=20,main=paste("p-Splines Residuals"),xlab="",ylab="Residuals")
plot(x=dat.x,y=as.numeric((pred.learn.quick-dat.y)),col="purple",lty=2,pch=20,main=paste("ANN Residuals"),xlab="",ylab="Residuals")
legend("topright", 95, legend=c("0","NEU-OLS","Loc-Reg"),
       col=c("seagreen1","red","blue"), pch=c(20,20,20), cex=0.75)
###
# PLOT FITS
#------------------#
# Writes data
l.inf.normalizacion<-function(x){x/(sqrt(sum(x^2)))}
dat.x.killa<-l.inf.normalizacion(x.dec.dat[,1])
dat.y.killa<-l.inf.normalizacion(x.dec.dat[,2]) # Wierd flip happens some times... so we flip it back...seems to be a coding kaka
# Plots
plot(dat.x,dat.y,pch=20,col=alpha("seagreen1",.5),type="o",main=paste(f.name),xlab="",ylab="")
lines(x=dat.x,y=dat.y.killa,type="l",col="red",lwd=3)
lines(x=dat.x,y=loc.fit,type="l",col="blue",lty=2)
lines(x=dat.x,y=p.fit,type="l",col="yellow",lty=2)
lines(x=dat.x,y=pred.learn.quick,type="l",col="purple",lwd=2)
lines(x=dat.x,y=dat.y.killa.loc,type="l",col="black",lwd=2)
legend("topleft", 95, legend=c("Rec. Dat.","NEU-OLS","Loc-Reg","p-Splines","ANN","NEU-Log-Reg"),
       col=c("seagreen1","red","blue","yellow","purple","black"), pch=c(20,20,20), cex=0.75)
###



# Amalgamaters Infos
#----------------------#
ESE<-c(EMSE.NEU.OLS,EMSE.NEU.loc,ESE.loc,ESE.p,EMSE.ANN)
#ESE<-c(as.vector(ESE),ESE.loc,ESE.p,EMSE.ANN)
Error.estimates<-rbind(mean.NEU.OLS.confid.interval,mean.NEU.OLS.confid.interval.loc,Error.estimates.loc,Error.estimates.p,ANN.MSE)

# Reports
#-----------------------------------#
ESE<-t(as.matrix(ESE))
colnames(ESE)<-c("NEU-OLS","NEU-LOESS","LOESS","p-Splines","ANN"); rownames(ESE)<-paste(y.noise)
rownames(Error.estimates)<-colnames(ESE);colnames(Error.estimates)<-c("95 L","Estimated Mean","95 U")
# Run-Time Facts
performance.facts<-t(as.matrix(c(as.numeric((NEU.OLS.end.time-NEU.OLS.start.time)),
                         as.numeric(loc.end.run-loc.start.run),
                     as.numeric(p.end.run-p.start.run),
                     as.numeric(ANN.end.run-ANN.start.run)
                         )))

rownames(performance.facts)<-paste(y.noise);colnames(performance.facts)<-c("NEU-OLS","Loc-Reg","p-Splines","ANN")

# Prints the Shints
#------------#------------#
###
if(FALSE){
cat("\014")
xtable(ESE,label="tab_ESEs_sim",caption="Estimated Mean Squared Error",digits = 3,display = c("s","e","e","e","e","e"))
xtable(Error.estimates,label="tab_EMEs_CI_sim",caption="Estimates of Confidence Intervals for Errors",digits = 3,display = c("s","e","e","e"))
xtable(performance.facts,label="tab_NEUOLS_perform_sim",caption="Runtime Metrics")
}
# Optimal Lambda
print(paste("Optimal Lambda=",round(test$cv,3)))

## Report
ESE
Error.estimates
performance.facts
