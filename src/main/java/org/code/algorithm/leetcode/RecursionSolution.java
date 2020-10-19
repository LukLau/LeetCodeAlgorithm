package org.code.algorithm.leetcode;

import org.code.algorithm.datastructe.ListNode;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;

/**
 * @author dora
 * @date 2020/10/8
 */
public class RecursionSolution {
    public static void main(String[] args) {
        RecursionSolution solution = new RecursionSolution();
        List<String> wordDict = Arrays.asList("a");
        solution.wordBreakV2("ab", wordDict);
    }


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
        if (s == null) {
            return true;
        }
        HashMap<String, Boolean> map = new HashMap<>();

        return intervalWordBreak(map, s, wordDict);
    }

    private boolean intervalWordBreak(HashMap<String, Boolean> map, String s, List<String> wordDict) {
        if (map.containsKey(s)) {
            return false;
        }
        if (s.isEmpty()) {
            return true;
        }
        for (String word : wordDict) {
            if (!s.startsWith(word)) {
                continue;
            }
            String tmp = s.substring(word.length());
            boolean wordBreak = intervalWordBreak(map, tmp, wordDict);

            if (wordBreak) {
                return true;
            }
            map.put(tmp, false);
        }
        return false;
    }


    /**
     * 140. Word Break II
     *
     * @param s
     * @param wordDict
     * @return
     */
    public List<String> wordBreakV2(String s, List<String> wordDict) {
        if (s == null || s.isEmpty()) {
            return new ArrayList<>();
        }
        HashMap<String, List<String>> map = new HashMap<>();
        return intervalWordBreakV2(map, s, wordDict);
    }

    private List<String> intervalWordBreakV2(HashMap<String, List<String>> map, String s, List<String> wordDict) {
        if (map.containsKey(s)) {
            return map.get(s);
        }
        List<String> result = new ArrayList<>();
        if (s.isEmpty()) {
            result.add(s);
            return result;
        }
        for (String word : wordDict) {
            boolean startsWith = s.startsWith(word);

            if (!startsWith) {
                continue;
            }
            List<String> wordBreakV2 = intervalWordBreakV2(map, s.substring(word.length()), wordDict);
            for (String tmp : wordBreakV2) {
                result.add(word + (tmp.isEmpty() ? "" : " " + tmp));
            }
        }
        map.put(s, result);
        return result;
    }


    /**
     * 148. Sort List
     *
     * @param head
     * @return
     */
    public ListNode sortList(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }
        ListNode slow = head;
        ListNode fast = head;
        while (fast.next != null && fast.next.next != null) {
            slow = slow.next;
            fast = fast.next.next;
        }
        ListNode fastHead = slow.next;

        slow.next = null;

        ListNode sortList1 = sortList(head);

        ListNode sortList2 = sortList(fastHead);

        return mergeList(sortList1, sortList2);
    }

    private ListNode mergeList(ListNode node1, ListNode node2) {
        if (node1 == null && node2 == null) {
            return null;
        }
        if (node1 == null) {
            return node2;
        }
        if (node2 == null) {
            return node1;
        }
        if (node1.val <= node2.val) {
            node1.next = mergeList(node1.next, node2);
            return node1;
        } else {
            node2.next = mergeList(node1, node2.next);
            return node2;
        }
    }


    // --波兰数系列- //

    /**
     * todo
     * 150. Evaluate Reverse Polish Notation
     *
     * @param tokens
     * @return
     */
    public int evalRPN(String[] tokens) {
        if (tokens == null || tokens.length == 0) {
            return 0;
        }
        List<Integer> signIndex = new ArrayList<>();
        for (String token : tokens) {

        }
        return -1;
    }


}
