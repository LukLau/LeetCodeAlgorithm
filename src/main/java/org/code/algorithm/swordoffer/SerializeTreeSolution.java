package org.code.algorithm.swordoffer;

import org.code.algorithm.datastructe.TreeNode;

import java.util.Deque;

/**
 * @author dora
 * @date 2020/8/3
 */
public class SerializeTreeSolution {


    private int index = -1;

    public static void main(String[] args) {
        TreeNode root = new TreeNode(0);
        root.left = new TreeNode(1);
        root.right = new TreeNode(2);
    }

    public String Serialize(TreeNode root) {
        StringBuilder builder = new StringBuilder();
        intervalSerialize(builder, root);
        return builder.toString();
    }

    private void intervalSerialize(StringBuilder builder, TreeNode root) {
        if (root == null) {
            builder.append("#,");
            return;
        }
        builder.append(root.val).append(",");
        intervalSerialize(builder, root.left);
        intervalSerialize(builder, root.right);
    }

    public TreeNode Deserialize(String str) {
        if (str == null || str.isEmpty()) {
            return null;
        }
        String[] words = str.split(",");
//        Deque<String> nodes = new LinkedList<>(Arrays.asList(words));
//        return intervalDeserialize(nodes);


        return intervalDeserialize(words);

    }

    private TreeNode intervalDeserialize(Deque<String> deque) {
        if (deque.isEmpty()) {
            return null;
        }
        String word = deque.pollFirst();
        if ("#".equals(word)) {
            return null;
        }
        TreeNode root = new TreeNode(Integer.parseInt(word));
        root.left = intervalDeserialize(deque);
        root.right = intervalDeserialize(deque);
        return root;
    }

    private TreeNode intervalDeserialize(String[] words) {
        index++;
        if (index >= words.length) {
            return null;
        }
        if ("#".equals(words[index])) {
            return null;
        }
        TreeNode root = new TreeNode(Integer.parseInt(words[index]));
        root.left = intervalDeserialize(words);
        root.right = intervalDeserialize(words);
        return root;

    }


}
