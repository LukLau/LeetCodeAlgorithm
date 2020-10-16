package org.code.algorithm.leetcode;

import java.util.ArrayList;
import java.util.List;

/**
 * @author dora
 * @date 2020/10/8
 */
public class RecursionSolution {


    /**
     * todo
     * 126. Word Ladder II
     *
     * @param beginWord
     * @param endWord
     * @param wordList
     * @return
     */
    public List<List<String>> findLadders(String beginWord, String endWord, List<String> wordList) {
        return null;
    }


    /**
     * todo
     * 127. Word Ladder
     *
     * @param beginWord
     * @param endWord
     * @param wordList
     * @return
     */
    public int ladderLength(String beginWord, String endWord, List<String> wordList) {
        return -1;
    }


    // ---深度优先遍历---//


    /**
     * https://leetcode.com/problems/surrounded-regions/
     * 130. Surrounded Regions
     *
     * @param board
     */
    public void solve(char[][] board) {
        if (board == null || board.length == 0) {
            return;
        }
        int row = board.length;
        int column = board[0].length;
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < column; j++) {
                boolean edgeScene = i == 0 || i == row - 1 || j == 0 || j == column - 1;
                if (edgeScene && board[i][j] == 'O') {
                    dfsReplace(i, j, board);
                }
            }
        }

        for (int i = 0; i < row; i++) {
            for (int j = 0; j < column; j++) {
                if (board[i][j] == 'o') {
                    board[i][j] = 'O';
                } else if (board[i][j] == 'O') {
                    board[i][j] = 'X';
                }
            }
        }


    }

    private void dfsReplace(int i, int j, char[][] board) {
        if (i < 0 || i >= board.length || j < 0 || j >= board[i].length) {
            return;
        }
        if (board[i][j] != 'O') {
            return;
        }
        board[i][j] = 'o';

        dfsReplace(i - 1, j, board);
        dfsReplace(i + 1, j, board);
        dfsReplace(i, j - 1, board);
        dfsReplace(i, j + 1, board);
    }


    /**
     * todo
     * 131. Palindrome Partitioning
     *
     * @param s
     * @return
     */
    public List<List<String>> partition(String s) {
        if (s == null || s.isEmpty()) {
            return new ArrayList<>();
        }
        List<List<String>> result = new ArrayList<>();
        intervalPartition(result, new ArrayList<>(), 0, s);
        return result;
    }

    private void intervalPartition(List<List<String>> result, List<String> tmp, int start, String s) {
        if (start == s.length()) {
            result.add(new ArrayList<>(tmp));
            return;
        }
        for (int i = start; i < s.length(); i++) {
            if (!partitionValid(s, start, i)) {
                continue;
            }
            tmp.add(s.substring(start, i + 1));
            intervalPartition(result, tmp, i + 1, s);
            tmp.remove(tmp.size() - 1);
        }
    }

    private boolean partitionValid(String s, int begin, int end) {
        if (begin == end) {
            return true;
        }
        while (begin < end) {
            if (s.charAt(begin) != s.charAt(end)) {
                return false;
            }
            begin++;
            end--;
        }
        return true;
    }


    /**
     * 139. Word Break
     *
     * @param s
     * @param wordDict
     * @return
     */
    public boolean wordBreak(String s, List<String> wordDict) {
        if (s == null || s.isEmpty()) {
            return true;
        }
        for (String word : wordDict) {
            s.startsWith(word);
        }
        return false;

    }


}
