package org.code.algorithm.leetcode;

import java.util.*;

/**
 * @author luk
 * @date 2020/11/19
 */
public class ForPage {


    /**
     * todo
     * 301. Remove Invalid Parentheses
     *
     * @param s
     * @return
     */
    public List<String> removeInvalidParentheses(String s) {
        if (s == null) {
            return new ArrayList<>();
        }
        List<String> result = new ArrayList<>();
        Set<String> visited = new HashSet<>();
        LinkedList<String> deque = new LinkedList<>();
        deque.offer(s);
        while (!deque.isEmpty()) {
            String poll = deque.poll();
            if (intervalInvalid(poll)) {
                result.add(poll);
            }
            if (!result.isEmpty()) {
                continue;
            }
            int len = poll.length();
            for (int i = 0; i < len; i++) {
                char word = s.charAt(i);
                if (word != '(' && word != ')') {
                    continue;
                }
                String tmp = poll.substring(0, i) + poll.substring(i + 1);

                if (!visited.contains(tmp)) {
                    visited.add(tmp);
                    deque.offer(tmp);
                }
            }
        }
        if (result.isEmpty()) {
            result.add("");
        }
        return result;
    }

    private boolean intervalInvalid(String poll) {
        if (poll == null || poll.isEmpty()) {
            return false;
        }
        int count = 0;
        char[] words = poll.toCharArray();
        for (char word : words) {
            if (word != '(' && word != ')') {
                continue;
            }
            if (word == '(') {
                count++;
            } else {
                if (count == 0) {
                    return false;
                }
                count--;
            }
        }
        return count == 0;
    }

    public static void main(String[] args) {
        ForPage page = new ForPage();
    }

}
