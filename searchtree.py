import types
from queue import PriorityQueue
from pydot import Dot, Node, Edge
from IPython.core.display import SVG


class SearchTree:

  def __init__(self, initial_state, goal_state, actions):
     self.initial_state = initial_state
     self.goal_state = goal_state
     self.actions = actions
     
  #--------------------------------------
  # Breadth First Search
  def bfs(self):

    stack, explored = [], set()
    stack.append(self.initial_state)
    graph = self._new_graph(str(self.initial_state))

    while stack:
      current_state = stack.pop(0)
      current_node = graph.get_node(f'"{current_state}"')[0]

      if current_state == self.goal_state:
        current_node.set_style("filled")
        current_node.set_fillcolor("green")
        return True, graph

      explored.add(current_state)
      known_nodes = explored | set(stack)
      any_child = False

      for action in self.actions:
        new_state = self._apply_action(current_state, action)

        if self._valid_state(new_state) and new_state not in known_nodes:
          any_child = True
          stack.append(new_state)
          known_nodes.add(new_state)
          new_node = Node(str(new_state))

          graph.add_node(new_node)
          graph.add_edge(Edge(current_node, new_node, fontsize=8, label=str(action)))
      
      if not any_child:
        current_node.set_style("filled")
        current_node.set_fillcolor("red")
          
    return False, graph
  
  #--------------------------------------
  # Depth First Search
  def dfs(self):
  
    def recursion(state, parent_state, act):
      node = Node(str(state))
      graph.add_node(node)

      if parent_state:
        parent_node = Node(str(parent_state))
        graph.add_edge(Edge(parent_node, node, fontsize=8, label=str(act)))

      if state == self.goal_state:
        node.set_style("filled")
        node.set_fillcolor("green")
        return True
      
      explored.add(state)
      any_child = False

      for action in self.actions:
        new_state = self._apply_action(state, action)

        if self._valid_state(new_state) and new_state not in explored:
          any_child = True
          if recursion(new_state, state, action):
            return True
      
      if not any_child:
        node.set_style("filled")
        node.set_fillcolor("red")
      
      return False

    #----------------
    explored = set()
    graph = self._new_graph()

    solution = recursion(self.initial_state, None, None)
    return solution, graph
  
  #--------------------------------------
  # Greedy Best-First Search
  def gbfs(self):
      queue = PriorityQueue()
      position = 0
      queue.put((0, position, self.initial_state))
      costs = {self.initial_state: 0}

      graph = self._new_graph(str(self.initial_state))

      while not queue.empty():
        _, _, current_state = queue.get()
        current_node = graph.get_node(f'"{current_state}"')[0]
        any_child = False

        if current_state == self.goal_state:
          current_node.set_style("filled")
          current_node.set_fillcolor("green")
          return True, graph

        for action in self.actions:
          new_state = self._apply_action(current_state, action)
          new_cost = costs[current_state] + 1

          if self._valid_state(new_state) and new_cost < costs.get(new_state, float('inf')):
            costs[new_state] = new_cost
            f = self.heuristic_gbfs(new_state, self.goal_state) # calculate f(n) = h(n)
            position += 1
            queue.put((f, position, new_state))

            any_child = True
            new_node = Node(str(new_state))
            graph.add_node(new_node)
            graph.add_edge(Edge(current_node, new_node, fontsize=8, label=f"{action} cost: {f}"))

        if not any_child:
          current_node.set_style("filled")
          current_node.set_fillcolor("red")

      return False, graph
  
  #--------------------------------------
  # A Star Search
  def a_star(self):
    queue = PriorityQueue()
    position = 0
    queue.put((0, position, self.initial_state))
    costs = {self.initial_state: 0}

    graph = self._new_graph(str(self.initial_state))

    while not queue.empty():
      _, _, current_state = queue.get()

      current_node = graph.get_node(f'"{current_state}"')[0]
        
      if current_state == self.goal_state:
        current_node.set_style("filled")
        current_node.set_fillcolor("green")
        return True, graph

      any_child = False

      for action in self.actions:
        new_state = self._apply_action(current_state, action)       
        new_cost = costs[current_state] + 1

        if self._valid_state(new_state) and new_cost < costs.get(new_state, float('inf')):
          costs[new_state] = new_cost
          
          f = new_cost + self.heuristic_astar(new_state) # calculate f(n) = g(n) + h(n)
          position += 1
          queue.put((f, position, new_state))
          any_child = True

          new_node = Node(str(new_state))
          graph.add_node(new_node)
          graph.add_edge(Edge(current_node, new_node, fontsize=8, label=f"{action} cost: {f}"))
      
      if not any_child:
        current_node.set_style("filled")
        current_node.set_fillcolor("red")

    return False, graph

  #--------------------------------------
  def heuristic_gbfs(self, state, goal_state):
    return sum(abs(a - b) for a, b in zip(state, goal_state))
  
  #--------------------------------------
  def heuristic_astar(self, state):
    m, c, _ = state  
    return m + c

  #--------------------------------------
  def _valid_state(self, state, total=3):
    m, c = state[:2]
    left_ok = m >= c or m == 0
    right_ok = total - m >= total - c or total - m == 0

    rg = range(total + 1)
    return (m in rg and c in rg) and left_ok and right_ok

  #--------------------------------------
  def _apply_action(self, state, action):
    left = bool(state[-1])
    op = -1 if left else 1
    new_state = (state[0] + op * action[0], state[1] + op * action[1], int(not left))
    return new_state
  
  #--------------------------------------
  def _new_graph(self, initial_label=""):
    graph = Dot( graph_type='graph', mode="wide")
    graph.show = types.MethodType(lambda self: SVG(self.create_svg()), graph)
    
    if initial_label:
      graph.add_node(Node(initial_label))

    return graph