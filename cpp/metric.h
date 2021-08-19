//
//  metric.h
//  cpp
//
//  Created by 陈一帆 on 2020/3/19.
//  Copyright © 2020 陈一帆. All rights reserved.
//

#ifndef metric_h
#define metric_h

extern "C"
void recall(float**score, long* cutoffs, float*res, long num_row, long num_col, long num_cuts){
    for (int i=0; i<num_row; i++) {
        float target_score = score[i][0];
        long rank = 0;
        for (int j=1; j<num_col; j++)
            if (target_score < score[i][j])
                rank++;
        
        for (int c=0; c<num_cuts; c++)
            if (rank < cutoffs[c])
                res[c] += 1;
    }
    
    for (int c=0; c<num_cuts; c++)
        res[c] /= num_row;
}

#endif /* metric_h */
