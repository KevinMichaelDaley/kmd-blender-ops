import bmesh, bpy
from mathutils import Matrix, Vector
import math
def get_active_vert(bm):
  """simple utility function to search the selection history for the 'active' vertex; note that this only means anything in blender
  terminology if the immediately last object selected was an vertex"""
    if bm.select_history:
        elem = bm.select_history[-1]
        if isinstance(elem, bmesh.types.BMVert):
            return elem
    return None

def get_active_edge(bm):
  """simple utility function to search the selection history for the 'active' edge; note that this only means anything in blender
  terminology if the immediately last object selected was an edge"""
    if bm.select_history:
        elem = bm.select_history[-1]
        if isinstance(elem, bmesh.types.BMEdge):
            return elem
    return None
def get_selected_loop_sorted(m):
  """get all the selected vertices in the currently selected edge loop (must be one contiguous edge loop with an active vertex or edge)
  and the edges between them and sort them so that the active vertex is first and all subsequent edges are in order by connectivity:
  *--*--*---*---*
  is sorted from left to right.
  If there is an edge from the last to the first vertex it is included at the end.
  """
    selected_verts=[]
    for v in m.verts:
        if v.select:
            selected_verts.append(v)
    N=len(selected_verts)
    edges={}
    next=[]
    for v in selected_verts:
       edges[v.index]=[]
    for v in selected_verts:
       for e in v.link_edges:
           if e in edges:
               continue
           v2=e.other_vert(v)
           if not v2.select:
              continue
           edges[v.index].append(e)
           edges[v2.index].append(e)
    v0=get_active_vert(m)
    loop_v=[v0]
    loop_e=[]
    all_done=False
    while len(loop_v)<len(selected_verts) and not all_done:
      v=loop_v[-1]
      elist=edges[v.index]
      all_done=True
      for e in elist:
        v2 = e.other_vert(v)
        if v2 not in loop_v:
           loop_v.append(v2)
           loop_e.append(e)
           all_done=False
           break
    for e in edges[loop_v[-1].index]:
       if e not in loop_e and loop_v[0] in e.verts:
            loop_e.append(e)
            break
    
    return loop_v, loop_e
    
def four_pole_junction(me):
  
  """common topology/edge-flow manipulation routine: turn
  *   --  *  --  *
  |              |
  *              *
  |              |
  |              |
  |              |
  * - * - * - *--*
  into
  *   --  *  --  *
  |     /   \    |
  * -- *     * --*
  |    |\   /|   |
  |    |  *  |   |
  |    |  |  |   |
  * -  *- *- *-- *
  without moving any of the existing vertices.
  Interior faces should be deleted first and the active vertex must be the existing vertex
  of the 'diamond' shape (center-of-top in the figure above).
  """
  
    m=bmesh.from_edit_mesh(me)
    loop_v, loop_e=get_selected_loop_sorted(m)
    if len(loop_v)!=10:
       return False
    u=(loop_v[2].co-loop_v[1].co)
    v=(loop_v[4].co-loop_v[5].co)
    p1=loop_v[0].co+u+v
    p3=loop_v[0].co+u-v
    p2=loop_v[0].co+1.5*u
    v1=m.verts.new(p1)
    v2=m.verts.new(p2)
    v3=m.verts.new(p3)
    for face in [[loop_v[0],v1,v2,v3],
      [loop_v[0],loop_v[1],loop_v[2],v1],
      [loop_v[0],v3,loop_v[8],loop_v[9]],
      [v1,loop_v[2],loop_v[3], loop_v[4]],
      [v3,loop_v[6],loop_v[7], loop_v[8]],
      [v1,loop_v[4],loop_v[5],v2],
      [v2,loop_v[5],loop_v[6],v3]]:
          f=m.faces.new(face[::-1])
    bmesh.update_edit_mesh(me)
    return True
def one_pole_junction(me):
  """common topology/edge-flow manipulation routine: turn
  * - * - *
  |       |
  *       |
  |       |
  * - - - *
  (with interior faces deleted and the bottom right corner active) into 
  * - * - *
  |   |   |
  * - *   |
  |     \ |
  * - - - *
  which results in a redirection of all loop cuts through the region.
  Supports any equivalent configuration of vertices and edges as input but the corner
  opposite the split edges should always be the active vertex.
  """
    m=bmesh.from_edit_mesh(me)      
    loop_v, loop_e=get_selected_loop_sorted(m)
    if len(loop_v)!=6:
        return False
    v1,v2,v3,v4,v5,v6=loop_v
    p=(v1.co+v2.co+v4.co+v6.co)/4.0
    v7=m.verts.new(p)
    f1=m.faces.new((v1,v2,v3,v7))
    f2=m.faces.new((v3,v4,v5,v7))        
    f3=m.faces.new((v7,v5,v6,v1))
    bmesh.update_edit_mesh(me)
    return True        
def project_onto_shapes(me_ob, axis='normal',scale_factor=1.0, max_distance=1.0, break_on_max_dist=True, invert=False):
  """this function projects the selected vertices along the 'axis' argument (which could be +/- X,Y,Z, the normal, or a vector
     towards the surface of all the other objects in the scene using a raycast.  It is quite handy for creating complex deformations.
     Arguments:
     'axis' is the vector you want to use as a ray from the current vertex.  This is used to compute the distance to the nearest point on the inactive scene objects.
     'scale_factor' is how far towards the scene you want to project, as a coefficient of the distance computed.  So if you want to project a grid /towards/ a sphere but not all the way onto it,
           you'd lower this value to, say, 0.5.
     'max_distance' is the maximum displacement to apply to any vertex.
     'break_on_max_dist' should be True if you want the vertex to stay where it is if the raycast into the scene doesn't hit anything.
     Otherwise it'll be displaced by 'max_distance' in the direction specified by your 'axis' argument.
     (the difference between this:
                *
                *
            *-- *
            * x x
            * x x  
            *---*
                *
                *
     and 
        *       
        * -----   
            *-- *
            * x x
            * x x  
            *---*
        *------
        *
     ).
     
    'invert': if set to True then the vertices are displaced in the opposite direction of the raycast, like a 'repulsion' effect.
              so they will conform to some sort of inside-out version of the scene.
    """
    world=[]
    for ob in bpy.context.scene.objects:
        if ob.type!='MESH':
            continue
        if ob is me_ob:
            continue
        world.append(ob)
    m=bmesh.from_edit_mesh(me_ob.data)
    B = me_ob.matrix_world
    for v in m.verts:
        if not v.select:
           continue
        n = v.normal
        d=max_distance
        if axis=='normal':
            axis_vector = B.to_3x3().normalized() @ n
        elif axis=='-normal':
            axis_vector = B @ (-n)
        elif axis=='Z':
            axis_vector = Vector((0,0,1))
        elif axis=='Y':
            axis_vector = Vector((0,1,0))
        elif axis=='X':
            axis_vector = Vector((1,0,0))
        elif axis=='-Z':
            axis_vector= Vector((0,0,-1))
        elif axis=='-Y':
            axis_vector=Vector((0,-1,0))
        elif axis=='-X':
            axis_vector=Vector((-1,0,0))
        else:
            axis_vector=axis
        for ob in world:
            B2 = ob.matrix_world
            B2inv = B2.inverted() 
            ray_origin = B2inv @ B @ v.co
            ray_dest = B2inv @ (B @ v.co +  axis_vector)
            ray_direction = (ray_dest - ray_origin).normalized()
            result = ob.ray_cast(ray_origin, ray_direction)
            if result[1] is not None:
                pos_hit=result[1]
                pos_hit = B.inverted() @ B2 @ pos_hit
                d2=(v.co-pos_hit).length
                d = min(d,d2)
        if d>max_distance-1e-7 and break_on_max_dist:
           d=0
        else:    
           v.co+=d*(B.inverted().to_3x3().normalized() @ axis_vector)*scale_factor*(1-2*invert)
    bmesh.update_edit_mesh(me_ob.data)
    return True        
    
       
def collapse_loop(me):
    """this function will turn this:
       * -- * -- *
       |    |    |
       * -- * -- *
       |    |    |
       * -- * -- *
       |    |    |
       * -- * -- *
       (with the vertices of the middle edges selected)
       into this:
       * --  -- *
       |        |
       * --  -- *
       |        |
       * --  -- *
       |        |
       * --  -- *
       ; i.e. it is an inverse-loop-cut.  It has only been tested on quads and only works when
       the edges selected could have been generated by a loop-cut operation; for example, none of the selected vertices
       should have more than 4 edges adjacent.
       """
    m=bmesh.from_edit_mesh(me)
    loop_v, loop_e=get_selected_loop_sorted(m)
    left_v={}
    right_v={}
    for i,v in enumerate(loop_v):
        for e in v.link_edges:
            if e not in loop_e:
              v2=e.other_vert(v)
              right=True
              if v in left_v:
                  right_v[v]=v2
                  break
              if len(left_v)==0:
                  left_v[v]=v2
                  continue
              for e in v2.link_edges:
                  if e in loop_e:
                      continue
                  if e.other_vert(v2) in left_v.values():
                      right=False
                      left_v[v]=v2
                      break
              if right:
                right_v[v]=v2
    v0=loop_v[-1]
    for v in loop_v:
        m.edges.new([left_v[v],right_v[v]])
        m.faces.new((left_v[v0],right_v[v0], right_v[v],left_v[v]))
        v0=v
    for v in loop_v:
        m.verts.remove(v)
    bmesh.update_edit_mesh(me)    
    return True
     
#project_onto_shapes(bpy.context.active_object,scale_factor=1, max_distance=10,
#        break_on_max_dist=False, axis='-Z', invert=False)
def count_vertices(me):
    m=bmesh.from_edit_mesh(me)
    sel_v = [v for v in m.verts if v.select]
    cuts=len(sel_v)
    return cuts
def roll(l,i):
    return list(l[i:])+list(l[:i])
def select_congruent(me, threshold=1e-4):
  """ This function (still being tested) selects every patch of faces congruent to the currently-selected face patch;
      it does not consider rotated or scaled versions as being congruent.  For now it only works with translated versions
      of the selection; it will also usually fail to capture congruences within a small tolerance due to extreme sensitivity."""
      
    m=bmesh.from_edit_mesh(me)
    sel_f = [f for f in m.faces if f.select]
    K=len(sel_f)
    congruences=[]
    def is_edge_congruent(e1,e2):
        v1=e1.verts[0].co-e1.verts[1].co
        v2=e2.verts[0].co-e2.verts[1].co
        if math.fabs(v1.dot(v2) - v2.dot(v2))<threshold:
            return True
        if math.fabs(v1.dot(v2) + v2.dot(v2))<threshold:
            return True
        return False
            

          
          
    def is_face_adjacency_congruent(f1,f2,s1):
        adjoining_edge=None
        for e in f1.edges:
            for e2 in f2.edges:
                if is_edge_congruent(e,e2):
                    adjoining_edge=e
                    break
        if adjoining_edge is None:
            return False
        for e3 in s.edges:
            if is_edge_congruent(adjoining_edge,e3):
                return True
        return False          
                
    for s in sel_f:      
      for f in m.faces:
        if f.select:
            continue
        for i in range(len(f.edges)):
            congruent=True
            edges=roll(f.edges,i)
            for j,e1 in enumerate(edges):
                e2=s.edges[j]
                if not is_edge_congruent(e1,e2):
                    congruent=False
            if congruent:
                congruences.append([[s,f]])
                break
              

    for s in sel_f:
      congruences2=[]
      for c in congruences:
            ss2=[x[0] for x in c]
            if s in ss2:
                continue
            ff2=[x[1] for x in c]
            for i,s2 in enumerate(ss2):
                if is_face_adjacency_congruent(s,ff2[i],s2):
                    c.append([s,ff2[i]])
     
    for c in congruences:
        if len(c)<len(sel_f):
            continue
        for x in c:
            x[1].select=True
    bmesh.update_edit_mesh(me)   
    return True
            
                
        
def align_loop(me, axis=Vector((0,1,0))):
    """Turns this:
       * -- * -- *
        \    \    \
         * -- * -- *
       into this:
       * -- * -- * 
       |    |    |
       * -- * -- *
       fixing the position of the active vertex.
       Will result in undefined behavior if applied to this:
       * -- * -- *
        \    \    \
         * -- * -- *
         /     \    \
         * ---  * -- *
       ; that is, it will only align pairs of vertices.
       Argument is the axis in local space along which the
       alignment should be performed per pair.
    """
    mask=Vector([1,1,1])-axis
    m=bmesh.from_edit_mesh(me)
    sel_v, sel_e = get_selected_loop_sorted(m)
    t1=sel_v[0].co-sel_v[1].co
    t2=sel_v[0].co-sel_v[-1].co
    v0=sel_v[0]
    if sel_v[-1]==v0:
        sel_v=sel_v[:-1]
    if abs(t2.dot(axis))>abs(t1.dot(axis)):
        sel_v=sel_v[::-1]
    len_over_2=len(sel_v)//2+1
    upper_half=sel_v[:len_over_2]
    lower_half=sel_v[len_over_2:]
    if v0 in lower_half:
        lower_half, upper_half = upper_half, lower_half
    if len(upper_half)!=len(lower_half):
        return False
    else:
        for i,v in enumerate(upper_half):
            v2=lower_half[-i]
            v2.co=v2.co+(v.co-v2.co).dot(axis)*axis
    bmesh.update_edit_mesh(me)
    return True           
               
#select_congruent(bpy.context.object.data)           
#align_loop(bpy.context.object.data,Vector((0,1,0)))
#collapse_loop(bpy.context.object.data)
#one_pole_junction(bpy.context.object.data)
print(count_vertices(bpy.context.object.data))
#align_loop(bpy,context.object.data,axis=Vecto
