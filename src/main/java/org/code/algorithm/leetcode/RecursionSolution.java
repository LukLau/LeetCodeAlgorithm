package org.code.algorithm.leetcode;

import org.code.algorithm.datastructe.ListNode;
import org.code.algorithm.datastructe.Trie;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Stack;
import java.util.stream.Collectors;

/**
 * @author dora
 * @date 2020/10/8
 */
public class RecursionSolution {
    public static void main(String[] args) {
        RecursionSolution solution = new RecursionSolution();
        char[][] board = new char[][]{{'o', 'a', 'a', 'n'}, {'e', 't', 'a', 'e'}, {'i', 'h', 'k', 'r'}, {'i', 'f', 'l', 'v'}};
        String[] words = new String[]{"oath", "pea", "eat", "rain"};

        String num = "105";
        solution.addOperators(num, 5);
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
                boolean edge = i == 0 || i == row - 1 || j == 0 || j == column - 1;
                if (edge && board[i][j] == 'O') {
                    dfsReplace(i, j, board);
                }
            }
        }
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < column; j++) {
                if (board[i][j] == '0') {
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
        board[i][j] = '0';
        dfsReplace(i - 1, j, board);
        dfsReplace(i + 1, j, board);
        dfsReplace(i, j - 1, board);
        dfsReplace(i, j + 1, board);
    }


    /**
     * todo
     * 200. Number of Islands
     *
     * @param grid
     * @return
     */
    public int numIslands(char[][] grid) {
        if (grid == null || grid.length == 0) {
            return 0;
        }
        int row = grid.length;
        int column = grid[0].length;
        boolean[][] used = new boolean[row][column];
        int count = 0;
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < column; j++) {
                if (!used[i][j] && grid[i][j] == '1') {
                    count++;
                    intervalNumIsLands(used, i, j, grid);
                }
            }
        }
        return count;
    }

    private void intervalNumIsLands(boolean[][] used, int i, int j, char[][] grid) {
        if (i < 0 || i == grid.length || j < 0 || j == grid[i].length || used[i][j] || grid[i][j] != '1') {
            return;
        }
        used[i][j] = true;
        intervalNumIsLands(used, i - 1, j, grid);
        intervalNumIsLands(used, i + 1, j, grid);
        intervalNumIsLands(used, i, j - 1, grid);
        intervalNumIsLands(used, i, j + 1, grid);
    }


    /**
     * 该题类似于 排列组合
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
        int len = s.length();
        if (start == len) {
            result.add(new ArrayList<>(tmp));
            return;
        }
        for (int i = start; i < len; i++) {
            if (!partitionValid(s, start, i)) {
                continue;
            }
            tmp.add(s.substring(start, i + 1));
            intervalPartition(result, tmp, i + 1, s);
            tmp.remove(tmp.size() - 1);
        }
    }

    private boolean partitionValid(String s, int begin, int end) {
        if (begin > end) {
            return false;
        }
        while (begin < end) {
            if (s.charAt(begin++) != s.charAt(end--)) {
                return false;
            }
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
            fast = fast.next.next;
            slow = slow.next;
        }
        ListNode split = slow.next;

        slow.next = null;

        ListNode list1 = sortList(head);
        ListNode list2 = sortList(split);

        return mergeList(list1, list2);

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

    /**
     * 282. Expression Add Operators
     *
     * @param num
     * @param target
     * @return
     */
    public List<String> addOperators(String num, int target) {
        if (num == null || num.isEmpty()) {
            return new ArrayList<>();
        }
        List<String> result = new ArrayList<>();
        intervalAddOperators(result, "", num, 0, 0, 0, target);
        return null;
    }

    private void intervalAddOperators(List<String> result, String s, String num, int pos, long eval, int target, int i) {
        if (pos == num.length() && eval == target) {
            result.add(s);
            return;
        }
        for (int i = pos; i < num.length(); i++) {
            String tmp = s.substring(pos, i + 1);
            long parse = Long.parseLong(tmp);
            if (i == 0) {

            } else {

                intervalAddOperators(result, s + "+" + tmp, num, i + 1, eval + parse, target, target);

                intervalAddOperators(result, s + "-" + tmp, num, i + 1, eval - parse, target, target);

                intervalAddOperators(result, s + "*" + tmp, num, i + 1, eval -);
            }

        }
    }


    // --波兰数系列- //

    /**
     * 150. Evaluate Reverse Polish Notation
     *
     * @param tokens
     * @return
     */
    public int evalRPN(String[] tokens) {
        if (tokens == null || tokens.length == 0) {
            return 0;
        }
        Stack<Integer> stack = new Stack<>();
        for (String token : tokens) {
            if ("+".equals(token)) {
                Integer first = stack.pop();
                Integer second = stack.pop();
                stack.push(first + second);
            } else if ("-".equals(token)) {
                Integer first = stack.pop();
                Integer second = stack.pop();
                stack.push(second - first);
            } else if ("*".equals(token)) {
                Integer first = stack.pop();

                Integer second = stack.pop();

                stack.push(first * second);
            } else if ("/".equals(token)) {

                Integer first = stack.pop();

                Integer second = stack.pop();

                stack.push(second / first);

            } else {
                stack.push(Integer.parseInt(token));
            }
        }
        return stack.pop();
    }


    /**
     * 206. Reverse Linked List
     *
     * @param head
     * @return
     */
    public ListNode reverseList(ListNode head) {
        if (head == null || head.next == null) {
            return head;
        }
        ListNode next = head.next;

        ListNode node = reverseList(next);

        next.next = head;

        head.next = null;

        return node;
    }

    // 深度优先遍历系列 //


    /**
     * 79. Word Search
     *
     * @param board
     * @param word
     * @return
     */
    public boolean exist(char[][] board, String word) {
        if (board == null || board.length == 0) {
            return false;
        }
        int row = board.length;
        int column = board[0].length;
        boolean[][] used = new boolean[row][column];
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < column; j++) {
                if (board[i][j] == word.charAt(0) && intervalExist(used, board, i, j, 0, word)) {
                    return true;
                }
            }
        }
        return false;
    }

    private boolean intervalExist(boolean[][] used, char[][] board, int i, int j, int k, String word) {
        if (k == word.length()) {
            return true;
        }
        if (i < 0 || i >= board.length || j < 0 || j >= board[i].length || word.charAt(k) != board[i][j] || used[i][j]) {
            return false;
        }
        used[i][j] = true;
        if (intervalExist(used, board, i - 1, j, k + 1, word) |
                intervalExist(used, board, i + 1, j, k + 1, word) ||
                intervalExist(used, board, i, j - 1, k + 1, word) ||
                intervalExist(used, board, i, j + 1, k + 1, word)) {
            return true;
        }
        used[i][j] = false;
        return false;
    }


    /**
     * 212. Word Search II
     *
     * @param board
     * @param words
     * @return
     */
    public List<String> findWords(char[][] board, String[] words) {
        if (board == null || board.length == 0 || words == null) {
            return new ArrayList<>();
        }
        int row = board.length;
        int column = board[0].length;
        boolean[][] used = new boolean[row][column];
        Trie trie = new Trie();
        for (String word : words) {
            trie.insert(word);
        }
        List<String> result = new ArrayList<>();
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < column; j++) {
                if (trie.startsWith(String.valueOf(board[i][j]))) {
                    intervalFindWords(result, trie, used, i, j, board, "");
                }
            }
        }
        result = result.stream().distinct().collect(Collectors.toList());
        return result;
    }

    private void intervalFindWords(List<String> result, Trie trie, boolean[][] used, int i, int j, char[][] board, String prefixWord) {
        if (i < 0 || i >= board.length || j < 0 || j >= board[j].length || used[i][j]) {
            return;
        }
        prefixWord += board[i][j];

        if (!trie.startsWith(prefixWord)) {
            return;
        }
        if (trie.search(prefixWord)) {
            result.add(prefixWord);
        }
        used[i][j] = true;
        intervalFindWords(result, trie, used, i - 1, j, board, prefixWord);
        intervalFindWords(result, trie, used, i + 1, j, board, prefixWord);
        intervalFindWords(result, trie, used, i, j - 1, board, prefixWord);
        intervalFindWords(result, trie, used, i, j + 1, board, prefixWord);
        used[i][j] = false;
    }


}
