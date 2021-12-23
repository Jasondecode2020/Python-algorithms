/**
 * Definition for singly-linked list.
 * function ListNode(val, next) {
 *     this.val = (val===undefined ? 0 : val)
 *     this.next = (next===undefined ? null : next)
 * }
 */
/**
 * @param {ListNode} l1
 * @param {ListNode} l2
 * @return {ListNode}
 */

// Input: head = [1,2,3,4,5,6,7,8,9,10,11,12,13], m = 2, n = 3
// Output: [1,2,6,7,11,12]

function ListNode(val, next) {
  this.val = val === undefined ? 0 : val;
  this.next = next === undefined ? null : next;
}

var deleteNodes = function (head, m, n) {
  return head;
};

let head;
let headNode = ListNode(1, head);
// let temp = headNode;
// for (let i = 2; i < 14; i++) {
//   let node = ListNode(i);
//   temp.next = node;
//   temp = temp.next;
// }

console.log(headNode.val);
