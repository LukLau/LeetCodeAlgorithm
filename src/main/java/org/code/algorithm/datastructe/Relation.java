package org.code.algorithm.datastructe;

/**
 * leet Code 277 Find the Celebrity
 *
 * @author luk
 * @date 2020/11/17
 */
public class Relation {


    /**
     * @param n a party with n people
     * @return the celebrity's label or -1
     */
    public int findCelebrity(int n) {
        if (n == 1) {
            return 0;
        }
        int candidate = 0;
        for (int i = 1; i < n; i++) {
            if (knows(candidate, i)) {
                candidate = i;
            }
        }
        for (int i = 0; i < n; i++) {
            if (i != candidate) {
                if (knows(candidate, i) || !knows(i, candidate)) {
                    return -1;
                }
            }
        }
        return candidate;
    }

    public boolean knows(int a, int b) {
        return true;
    }
}
