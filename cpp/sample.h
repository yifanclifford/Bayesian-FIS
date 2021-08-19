//
//  sample.h
//  cpp
//
//  Created by 陈一帆 on 2020/3/18.
//  Copyright © 2020 陈一帆. All rights reserved.
//

#ifndef sample_h
#define sample_h

#include <stdio.h>
#include <assert.h>
#include <time.h>
#include <stdlib.h>
#include <map>
#include <set>
#include <vector>
#include <string>
#include <string.h>

using namespace std;

std::string inPath = "/Users/chenyifan/experiment/justice/dataset/mf/";
std::map<long, std::set<long>> pos_items;

long num_user;
long num_item;

struct Parameter {
    long *neg_p;
    long idx;
    long id;
};

void setInPath(char *path) {
    size_t len = strlen(path);
    inPath = "";
    for (long i = 0; i < len; i++)
        inPath = inPath + path[i];
    printf("Input Files Path : %s\n", inPath.c_str());
}

void read_positive() {
    FILE *fin;
    long tmp;
    float val;
    long num_pos, item, user;
    fin = fopen((inPath).c_str(), "r");
    char *line = NULL;
    size_t len = 0;
    getline(&line, &len, fin);
    getline(&line, &len, fin);
    tmp = fscanf(fin, "%ld %ld %ld\n", &num_user, &num_item, &num_pos);
    printf("%ld %ld %ld\n", num_user, num_item, num_pos);
    for (long i = 0; i < num_pos; i++) {
        tmp = fscanf(fin, "%ld %ld %f\n", &user, &item, &val);
        user--;
        item--;
        if (pos_items.find(user) == pos_items.end())
            pos_items[user] = std::set<long>();
        pos_items[user].insert(item);
    }
    fclose(fin);
//    printf("%ld %ld %ld\n", num_user, num_item, num_pos);
}

void print(std::vector<long> &input) {
    for (std::vector<long>::iterator iter = input.begin(); iter < input.end(); iter++)
        printf("%ld ", *iter);
    printf("\n");
}

extern "C"
long get_num_item() {
    return num_item;
}

extern "C"
void init(char *path) {
    srand(time(0));
    setInPath(path);
    read_positive();
    printf("finish initialization!\n");
}

extern "C"
long get_num_positive(long user) {
    return pos_items[user].size();
}

extern "C"
long get_num_negative(long user) {
    return num_item - get_num_positive(user);
}

extern "C"
void get_positive(long *items, long user) {
    std::set<long> &pos_item = pos_items[user];
    long idx = 0;
    for (long item: pos_item)
        items[idx++] = item;
}

extern "C"
void sample_negative(long *neg_p, long user, long negative) {
    std::set<long> &pos_item = pos_items[user];
    long i = 0;
    while (i < negative) {
        long item = rand() % num_item;
        if (pos_item.find(item) == pos_item.end())
            neg_p[i++] = item;
    }
}


extern "C"
void all_negative(long *neg_p, long user) {
    std::set<long> &pos_item = pos_items[user];
    long id = 0;
    for (int i = 0; i < num_item; i++) {
        if (i != neg_p[0] && pos_item.find(i) == pos_item.end())
            neg_p[id++] = i;
    }
}

#endif /* sample_h */
