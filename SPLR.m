%{
ʹ���µ����ݼ�3589*457��ʵ��Ԥ���ǩ��Ϊ���ƾ���ʹ����ʵ��ǩ��Ӧ��Ԥ�����Ϊ����������ֵ
�ٰ���ԭ�Ĺ�ʽ�������ϣ�kӦ�����ŵ����������Ӷ��𽥼�С������ʹ���������������࣬��ˣ�u�������1
��ʹ��ʵ�ʱ�ǩ��������֮��������ԣ�����Ϊ1������Ϊ0�����ƾ���ֻ��ͬһ���ʱֵ��Ϊ1�����¾������ϡ�裬��������ʵW���㣨����ʹ������ԭʼ�����ռ��������֮��������ԣ�
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
    %Initialize W and S����ʼ��������Ȩ�ؾ�������������Ծ���
%     W=ones(d,7);
    W=unifrnd (0,1,d,7);
    label=load("E:\SPLR\valid_true_lab.csv");%��֤��ʵ�ʱ�ǩ,������
    
    %calculate the similarity between samples
    Z=zeros(n,n);%����ʵ�ʱ�ǩ��������֮���������
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
        fprintf("����������%d,����������%d\n",num,iter);
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