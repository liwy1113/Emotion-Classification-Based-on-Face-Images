library (RColorBrewer) 
showMatrix<-function(x,...)
  image(t(x[nrow(x) :1 ,]) , xaxt = "none"  , yaxt = "none"  ,
        col = rev(colorRampPalette(brewer.pal(9,"Greys"))(100)), ...)

tu = read.csv('../Desktop/statistics/208/project/train.csv', header = FALSE)

newtu = sapply(1:(nrow(tu)/ncol(tu)), function(i) c(as.matrix(tu[(1:350)+350*(i-1),])))
x = t(newtu)

la = read.table("../Desktop/statistics/208/project/trainlabel.txt")
la = la[,1]

############PCA
x.new = x[-which(la == 2|la == 4),]
x.new = x.new[-601,]

id<-la[-which(la == 2|la == 4)]
id = id[-601]
faces.data.frame<-data.frame(cbind(id=id,x.new))

xc<-scale(faces.data.frame[,-1],scale=FALSE)
A<-t(xc)/sqrt(610-1)

A.egn<-eigen(t(A)%*%A)
pc<-A%*%A.egn$vectors
pc<-apply(pc,2,function(i)i/sqrt(sum(i*i)))
n<-100
sum(A.egn$value[1:n])/sum(A.egn$value)
pcs<-pc[,1:n]
yt<-xc%*%pcs

mu<-apply(faces.data.frame,2,mean)
mu = mu[-1]

xr<-yt%*%t(pcs)+matrix(mu,nrow=610,ncol=122500,byrow=TRUE)

#the eighth fitted image
x8<-matrix(x.new[8,],350,350)
x8r<-matrix(xr[8,],350,350)

par(mfrow = c(1,1), mar = c(0, 0, 0, 0))
showMatrix(matrix(x[8,],350,350))
showMatrix(x8)
showMatrix(x8r) 
showMatrix(matrix(mu,350,350))
showMatrix(matrix(pcs[,1],350,350))
showMatrix(matrix(pcs[,2],350,350))
showMatrix(matrix(pcs[,100],350,350))

#################clustering
#################one person
one = read.csv('../Desktop/statistics/208/project/cluster_data.csv', header = FALSE)

one = sapply(1:(nrow(one)/ncol(one)), function(i) c(as.matrix(one[(1:350)+350*(i-1),])))
x = t(one)
x = x[c(1,2,13,5:7,9:11,20:22),]

xd<-dist(x)
#par(mfrow=c(2,1))
par(mfrow = c(1,1), mar = c(5, 4, 4, 2) + 0.1)

#Hierachical clustering
#ptm <- proc.time()
hh<-hclust(xd,method="average")
#proc.time() - ptm

plot(hh,hang=-1)
library(TeachingDemos)
rect.hclust(hh,k=4)

for(i in 1:nrow(xs))
  subplot(showMatrix(matrix(xs[hh$order[i],],350,350)),i,70,size=c(0.4,0.4))
#perform minimax linkage clustering by Bien, J., and Tibshirani, R. (2011).
#install.packages("protoclust")
library(protoclust)

hm<-protoclust(xd)
plot (hm, hang=-1)
rect.hclust(hm,k=4)
for(i in 1:nrow(xs))
  subplot(showMatrix(matrix(xs[hm$order[i],],350,350)),i,70,size=c(0.6,0.6))
#single
hs<-hclust(xd,method="single")
plot(hs,hang=-1)
rect.hclust(hs,k=4)
for(i in 1:nrow(xs))
  subplot(showMatrix(matrix(xs[hs$order[i],],350,350)),i,70,size=c(0.6,0.6))
#complete
hc<-hclust(xd,method="complete")
plot(hc,hang=-1)
rect.hclust(hc,k=4)
for(i in 1:nrow(xs))
  subplot(showMatrix(matrix(xs[hc$order[i],],350,350)),i,70,size=c(0.6,0.6))
#centroid
hce<-hclust(xd,method="centroid")
plot(hce,hang=-1)
rect.hclust(hce,k=4)
for(i in 1:nrow(xs))
  subplot(showMatrix(matrix(xs[hce$order[i],],350,350)),i,70,size=c(0.6,0.6))
#average solution
cut1<-cutree(hh, k=4)
lapply(1:4, function(nc) row.names(xs)[cut1==nc]) 

#minimax solution
cut2<-cutree(hm, k=4)
lapply(1:4, function(nc) row.names(xs)[cut2==nc]) 

#single solution
cut3<-cutree(hs, k=4)
lapply(1:4, function(nc) row.names(xs)[cut3==nc]) 

#complete solution
cut4<-cutree(hc, k=4)
lapply(1:4, function(nc) row.names(xs)[cut4==nc]) 

#centroid solution
cut5<-cutree(hce, k=4)
lapply(1:4, function(nc) row.names(xs)[cut5==nc]) 

cars.com.d<-as.dendrogram(hh)
plot(cars.com.d)
rect.hclust(hh,k=4)
for(i in 1:nrow(xs))
  subplot(showMatrix(matrix(xs[hh$order[i],],350,350)),i,70,size=c(0.5,0.5))



library(cluster)
dv <- diana(xs, stand = TRUE)
print(dv)
plot(dv,which=2,hang=-1)
rect.hclust(dv,k=4)
for(i in 1:nrow(xs))
  subplot(showMatrix(matrix(xs[dv$order[i],],350,350)),i,10,size=c(0.6,0.6))

#K-means
samp.range <- function(x){
  myrange <- diff(range(x))
  return(myrange)
}
my.ranges <- apply(xs,2,samp.range)
xstd <- sweep(xs,2,my.ranges,FUN="/") 

xk3 <- kmeans(xs, centers=4, iter.max=100, nstart=25)
xk3
lapply(1:4, function(nc) row.names(xs)[xk3$cluster==nc])  

library(cluster)
x.kmed.3 <- pam(xs, k=4, diss=F)
lapply(1:4, function(nc) row.names(xs)[x.kmed.3$cluster==nc])  
x.kmed.3$silinfo$avg.width

my.k.choices <- 2:8
avg.sil.width <- rep(0, times=length(my.k.choices))
for (ii in (1:length(my.k.choices)) ){
  avg.sil.width[ii] <- pam(xs, k=my.k.choices[ii])$silinfo$avg.width
}
plot( cbind(my.k.choices, avg.sil.width),type='o' )
lapply(1:4, function(nc) row.names(xs)[x.kmed.3$clustering==nc])  
names(x.kmed.3)
names(xk3)
x.kmed.3$silinfo$avg.width
#spectral
library(kernlab)
sc<-specc(xs,centers=4)

lapply(1:4, function(nc) row.names(xs)[sc==nc]) 

summary(silhouette(cutree(hh, k=4), xd))$avg.width
summary(silhouette(cutree(hm, k=4), xd))$avg.width
summary(silhouette(cutree(hs, k=4), xd))$avg.width
summary(silhouette(cutree(hc, k=4), xd))$avg.width
summary(silhouette(cutree(hce, k=4), xd))$avg.width

for (ii in (1:length(my.k.choices)) ){
  avg.sil.width[ii] <- summary(silhouette(cutree(hh, k=my.k.choices[ii]), xd))$avg.width
}
print( cbind(my.k.choices, avg.sil.width) )

summary(silhouette(kmeans(xs, centers=4, iter.max=100, nstart=25)$cluster, xd))$avg.width

library(clusterCrit)
library(fossil)
rand.index(xk3$clus,rep(1:4,rep(3,4)))
adj.rand.index(xk3$clus,rep(1:4,rep(3,4)))

##############another person
two = read.csv('../Desktop/statistics/208/project/cluster_data2.csv', header = FALSE)

two = sapply(1:(nrow(two)/ncol(two)), function(i) c(as.matrix(two[(1:350)+350*(i-1),])))
x1 = t(two)
x1 = x1[c(1,6:8,16:18,23:25),]

xd<-dist(x1)
#par(mfrow=c(2,1))
par(mfrow = c(1,1), mar = c(5, 4, 4, 2) + 0.1)

#Hierachical clustering
#ptm <- proc.time()
hh<-hclust(xd,method="average")
#proc.time() - ptm

plot(hh,hang=-1)
library(TeachingDemos)
rect.hclust(hh,k=4)

for(i in 1:nrow(xs))
  subplot(showMatrix(matrix(xs[hh$order[i],],350,350)),i,70,size=c(0.4,0.4))

#########two person
xs = rbind(x,x1)

xs = x[c(1,2,13,5:7,9:11,20:22),]


xd<-dist(xs)
#par(mfrow=c(2,1))
par(mfrow = c(1,1), mar = c(5, 4, 4, 2) + 0.1)

#Hierachical clustering
#ptm <- proc.time()
hh<-hclust(xd,method="average")
#proc.time() - ptm

plot(hh,hang=-1)
library(TeachingDemos)
rect.hclust(hh,k=4)

for(i in 1:nrow(xs))
  subplot(showMatrix(matrix(xs[hh$order[i],],350,350)),i,70,size=c(0.4,0.4))



