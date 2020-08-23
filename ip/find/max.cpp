#include "max.h"

void max(DTYPE A[SIZE] , DTYPE* max ){
    DTYPE temp[STAGE][BATCH/2];
    DTYPE max_value = A[0];
#pragma HLS ARRAY_PARTITION variable=temp complete
#pragma HLS ARRAY_PARTITION variable=A complete

    for (int i=0; i<ITERATION; i++){
        #pragma HLS pipeline
        for(int b=0;b<BATCH;b=b+2){ // BATCH = 2^n
        	if( i*BATCH+b+1 <SIZE ){
            temp[0][b/2] = max_unit(A[i*BATCH+b],A[i*BATCH+b+1]);
        	}else if(i*BATCH+b<SIZE && i*BATCH+b+1>=SIZE ){
        	temp[0][b/2] = max_unit(A[i*BATCH+b],0);
        	}else {
        		temp[0][b/2] = 0 ;
        	}
        }

        for (int s=0; s<STAGE-1; s++){
            for(int m=0; m<BATCH/2; m=m+2){
                temp[s+1][m/2]= max_unit(temp[s][m],temp[s][m+1]);
        }    }
        
        if(temp[STAGE-1][0]>max_value)
            max_value = temp[STAGE-1][0];
        if(temp[STAGE-1][1]>max_value)
            max_value = temp[STAGE-1][1];
    }        
    *max= max_value;
}

DTYPE max_unit(DTYPE a , DTYPE b){
    return a>b?a:b ;
}



{

#define NUM 8

int sum = 0;
for (int i=0; i<NUM; i++){
    sum += A[i];
}


}
{


{{dtype}} max_{{kernel_size}}(
            {% for i in range(kernel_size-1) %}{{dtype}} a_0_{{i}},
            {% endfor %}{{dtype}} a_0_{{kernel_size-1}})
{
      {% for i in range(1,log2_kernel_size+1) %}{{dtype}} {% for j in range(((kernel_size//(2**i)))-1)%}a_{{i}}_{{j}},{% endfor %}a_{{i}}_{{ (kernel_size//(2**i))-1}}; 
      {% endfor %}{% for i in range(1,log2_kernel_size+1) %}{% for j in range(((kernel_size//(2**i))))%}
      if (a_{{i-1}}_{{j*2}} > a_{{i-1}}_{{j*2+1}}){a_{{i}}_{{j}} = a_{{i-1}}_{{j*2}};}
          else {a_{{i}}_{{j}} = a_{{i-1}}_{{j*2+1}};}{% endfor %}
      {% endfor %}
      return a_{{log2_kernel_size}}_0;
      
      




}