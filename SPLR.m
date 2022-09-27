%{
使用新的数据集3589*457，实际预测标签作为相似矩阵，使用真实标签对应的预测概率为分类器特征值
①按照原文公式，理论上，k应该随着迭代次数增加而逐渐减小，才能使简单样本数量逐渐增多，因此，u必须大于1
②使用实际标签度量样本之间的相似性，相似为1，否则为0，相似矩阵只有同一类别时值才为1，导致矩阵过于稀疏，不利于真实W计算（考虑使用样本原始特征空间计算样本之间的相似性）
input:
X:data matrix with n rows and d columns
alpha,lambda1:balance parameters
beta,k,mu:parameters related to self-paced learning
maxIter:maximum iteration number
output:
W:projection matrix with d rows
index:sorted index of the l2-norm of rows of matrix W
obj:values of the objective function during the iteration
%}
function [W,index,obj] = SPLR(X,Y,alpha,lambda1,beta,mu,maxIter)
    [n,d]=size(X);
    %Initialize W and S，初始化分类器权重矩阵和样本相似性矩阵
%     W=ones(d,7);
    W=unifrnd (0,1,d,7);
    label=load("E:\SPLR\valid_true_lab.csv");%验证集实际标签,列向量
    
    %calculate the similarity between samples
    Z=zeros(n,n);%根据实际标签计算样本之间的相似性
    for i=1:n
        for j=i:n
            if(label(1,i)==label(1,j))
                Z(i,j)=1;
                Z(j,i)=Z(i,j);
            end
        end
    end
    %calculate the Laplacian matrix L
    L=diag(sum(Z,2))-Z;
    iter=1;
    %Initialize k
    k=initializek(X,Y,W);    
    while (iter<=maxIter)
        %update v
        v=updateV(X,Y,W,k,beta);
        num=length(find(v~=0));
        fprintf("样本数量：%d,迭代次数：%d\n",num,iter);
        %update G
        G=diag(sqrt(v))*X;
        %update Q
        Q=diag(sqrt(v))*Y;
        %update V
        V=updateU(W);    
        %update W
        W=inv(G'*G+2*lambda1*X'*L*X+2*alpha*V)*G'*Q;     
        %calculate the value of the objective function
        obj(iter)=trace((Q-G*W)*(Q-G*W)')+alpha*trace(W'*updateU(W)*W)+beta^2/sum(v+beta*k)+lambda1*trace(W'*X'*L*X*W);
        %update k
        k=k/mu;
        iter=iter+1;
    end 
    %calculate the score
    score=sum((W.*W),2);
    [~,index]=sort(score,'descend');
end 