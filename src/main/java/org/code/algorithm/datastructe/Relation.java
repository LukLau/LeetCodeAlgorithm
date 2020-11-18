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
            return n;
        }
        int candidate = 0;
        for (int i = 0; i < n - 1; i++) {
            if (knows(candidate, i)) {
                candidate = i;
            }
        }
        for (int i = candidate + 1; i < n; i++) {
            if (knows(candidate, i)) {
                return -1;
            }
        }
        for (int i = 0; i < candidate; i++) {
            if (!knows(i, candidate) || knows(candidate, i)) {
                return -1;
            }
        }
        return candidate;
    }

    public boolean knows(int a, int b) {
        return true;
    }
}
