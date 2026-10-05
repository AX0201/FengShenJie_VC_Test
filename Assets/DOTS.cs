// 单文件 DOTS 2D 模拟
// 挂载在摄像机上使用(空场景,只有摄像机/UI/输入系统)。
// 1000x1000 草地网格 + 1万建筑(每2秒产1食物,上限100) + 1万AI(寻路/阻挡/攻击/进食) + WASD玩家 + 摄像机锁定跟随/滚轮缩放。
// 逻辑:Entities ECS + Burst;渲染:GPU 实例化(不依赖 Entities Graphics);监测:UI Toolkit 半透明曲线面板。

using System.Collections;
using System.Collections.Generic;
using System.Text;
using Unity.Burst;
using Unity.Collections;
using Unity.Entities;
using Unity.Jobs;
using Unity.Mathematics;
using Unity.Transforms;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.U2D;
using UnityEngine.UIElements;
#if ENABLE_INPUT_SYSTEM
using UnityEngine.InputSystem;
using UnityEngine.InputSystem.Controls;
using UnityEngine.InputSystem.LowLevel;
#endif

namespace DotsSim
{
    using Random = Unity.Mathematics.Random;

    // ========================= 常量 =========================

    public static class DotsConst
    {
        public const int MapSize = 1000;           // 地图边长(格)
        public const int CellSize = 32;            // 建筑空间哈希桶边长(格)
        public const int CellsPerRow = (MapSize + CellSize - 1) / CellSize; // 32

        public const float FoodInterval = 2f;      // 每2秒产1食物
        public const int FoodCap = 100;            // 食物上限
        public const int FoodLevels = 21;          // 建筑颜色深浅档位(每5食物一档)

        public const int BuildingHp = 3;
        public const int NpcHp = 10;
        public const int AttackDamage = 1;
        public const float AttackInterval = 1f;    // 被困时每秒攻击一次
        public const float ShakeTime = 0.25f;      // 攻击时角色晃动时长

        public const int DetourRadius = 5;         // 绕行搜索半径(格)
        public const int DetourSpan = DetourRadius * 2 + 1;

        public static int CellHash(int cx, int cy) => cy * CellsPerRow + cx;
        public static int CellIndex(int2 c) => c.y * MapSize + c.x;
        public static bool InMap(int2 c) => c.x >= 0 && c.y >= 0 && c.x < MapSize && c.y < MapSize;
        public static float2 Center(int2 c) => new float2(c.x + 0.5f, c.y + 0.5f);
        public static int BucketHash(int2 cell) => CellHash(
            math.clamp(cell.x / CellSize, 0, CellsPerRow - 1),
            math.clamp(cell.y / CellSize, 0, CellsPerRow - 1));
    }

    // ========================= ECS 组件 =========================

    /// <summary>建筑:食物存量、生产计时、生命、所在格。</summary>
    public struct BuildingData : IComponentData
    {
        public int Food;
        public float Timer;
        public int Hp;
        public int2 Cell;
    }

    public static class NpcState
    {
        public const byte Seek = 0;   // 寻找建筑
        public const byte Move = 1;   // 前往建筑
        public const byte Wait = 2;   // 进食后停顿
    }

    /// <summary>AI:目标建筑、所在格、状态、计时器、绕行路径进度、独立随机数。</summary>
    public struct NpcData : IComponentData
    {
        public Entity Target;
        public Entity Last;
        public int2 TargetCell;
        public int2 Cell;
        public byte State;
        public float WaitTimer;
        public float RethinkTimer;
        public float ShakeTimer;
        public float Speed;
        public int PathIndex;
        public Random Rng;
    }

    /// <summary>AI 生命单独成组件,便于在遍历 NpcData 的同时通过 Lookup 攻击其他 AI。</summary>
    public struct NpcHealth : IComponentData
    {
        public int Hp;
    }

    /// <summary>绕行路径(半径5格内 BFS 得到的格序列)。</summary>
    [InternalBufferCapacity(16)]
    public struct PathCell : IBufferElementData
    {
        public int2 Cell;
    }

    public struct PlayerTag : IComponentData { }

    /// <summary>单例:由 MonoBehaviour 每帧写入的玩家输入。</summary>
    public struct PlayerInputData : IComponentData
    {
        public float2 Move;
        public float Speed;
    }

    /// <summary>单例:占用网格(建筑与AI都占格,Entity.Null 为空地) + 建筑空间哈希。</summary>
    public struct GridSingleton : IComponentData
    {
        public NativeArray<Entity> Occupancy;
        public NativeParallelMultiHashMap<int, Entity> BuildingMap;
    }

    // ========================= 系统:建筑生产 =========================

    [BurstCompile]
    public partial struct FoodProductionSystem : ISystem
    {
        [BurstCompile]
        public void OnCreate(ref SystemState state)
        {
            state.RequireForUpdate<BuildingData>();
        }

        [BurstCompile]
        public void OnUpdate(ref SystemState state)
        {
            new FoodProduceJob { Dt = SystemAPI.Time.DeltaTime }.ScheduleParallel();
        }
    }

    [BurstCompile]
    public partial struct FoodProduceJob : IJobEntity
    {
        public float Dt;

        void Execute(ref BuildingData b)
        {
            if (b.Food >= DotsConst.FoodCap)
            {
                b.Timer = 0f; // 满仓停产,计时不累积
                return;
            }
            b.Timer += Dt;
            while (b.Timer >= DotsConst.FoodInterval && b.Food < DotsConst.FoodCap)
            {
                b.Timer -= DotsConst.FoodInterval;
                b.Food++;
            }
            if (b.Food >= DotsConst.FoodCap) b.Timer = 0f;
        }
    }

    // ========================= 系统:AI 决策 + 寻路 + 战斗 =========================

    [BurstCompile]
    [UpdateAfter(typeof(FoodProductionSystem))]
    public partial struct NpcBehaviorSystem : ISystem
    {
        NativeArray<int> _parent;
        NativeArray<int> _queue;

        [BurstCompile]
        public void OnCreate(ref SystemState state)
        {
            state.RequireForUpdate<GridSingleton>();
            state.RequireForUpdate<NpcData>();
            state.RequireForUpdate<EndSimulationEntityCommandBufferSystem.Singleton>();
            int n = DotsConst.DetourSpan * DotsConst.DetourSpan;
            _parent = new NativeArray<int>(n, Allocator.Persistent);
            _queue = new NativeArray<int>(n, Allocator.Persistent);
        }

        [BurstCompile]
        public void OnDestroy(ref SystemState state)
        {
            if (_parent.IsCreated) _parent.Dispose();
            if (_queue.IsCreated) _queue.Dispose();
        }

        [BurstCompile]
        public void OnUpdate(ref SystemState state)
        {
            var grid = SystemAPI.GetSingleton<GridSingleton>();
            var ecb = SystemAPI.GetSingleton<EndSimulationEntityCommandBufferSystem.Singleton>()
                .CreateCommandBuffer(state.WorldUnmanaged);

            // 单线程 Burst:AI 之间通过占用网格互相阻挡,串行处理才能保证"一格一人"
            new NpcThinkJob
            {
                Dt = SystemAPI.Time.DeltaTime,
                Occupancy = grid.Occupancy,
                BuildingMap = grid.BuildingMap,
                Buildings = SystemAPI.GetComponentLookup<BuildingData>(false),
                Health = SystemAPI.GetComponentLookup<NpcHealth>(false),
                Ecb = ecb,
                Parent = _parent,
                Queue = _queue
            }.Schedule();
        }
    }

    [BurstCompile]
    public partial struct NpcThinkJob : IJobEntity
    {
        public float Dt;
        public NativeArray<Entity> Occupancy;
        [ReadOnly] public NativeParallelMultiHashMap<int, Entity> BuildingMap;
        public ComponentLookup<BuildingData> Buildings;
        public ComponentLookup<NpcHealth> Health;
        public EntityCommandBuffer Ecb;
        public NativeArray<int> Parent;
        public NativeArray<int> Queue;

        void Execute(Entity self, ref NpcData npc, ref LocalTransform lt, DynamicBuffer<PathCell> path)
        {
            if (Health[self].Hp <= 0) return; // 本帧已被打死,等待 ECB 销毁

            npc.ShakeTimer = math.max(0f, npc.ShakeTimer - Dt);
            npc.RethinkTimer = math.max(0f, npc.RethinkTimer - Dt);

            if (npc.State == NpcState.Wait)
            {
                npc.WaitTimer -= Dt;
                if (npc.WaitTimer <= 0f) npc.State = NpcState.Seek;
                return;
            }

            if (npc.State == NpcState.Seek)
            {
                Seek(ref npc, path);
                if (npc.State != NpcState.Move) return;
            }

            if (!IsAliveBuilding(npc.Target))
            {
                npc.State = NpcState.Seek;
                ClearPath(ref npc, path);
                return;
            }

            int2 d = math.abs(npc.Cell - npc.TargetCell);
            if (math.max(d.x, d.y) <= 1)
            {
                Arrive(ref npc, path);
                return;
            }

            Step(self, ref npc, ref lt, path);
        }

        bool IsAliveBuilding(Entity e) => Buildings.HasComponent(e) && Buildings[e].Hp > 0;

        static void ClearPath(ref NpcData npc, DynamicBuffer<PathCell> path)
        {
            path.Clear();
            npc.PathIndex = 0;
        }

        /// <summary>抵达(与建筑相邻):吃掉一半食物(向上取整),停顿"吃掉数量"秒。</summary>
        void Arrive(ref NpcData npc, DynamicBuffer<PathCell> path)
        {
            var b = Buildings[npc.Target];
            int eat = (b.Food + 1) / 2;
            b.Food -= eat;
            Buildings[npc.Target] = b;

            npc.Last = npc.Target;
            npc.Target = Entity.Null;
            npc.State = NpcState.Wait;
            npc.WaitTimer = eat;
            ClearPath(ref npc, path);
        }

        // ---------- 寻找建筑:从附近桶逐环扩大,随机挑一个(排除刚吃过的) ----------

        void Seek(ref NpcData npc, DynamicBuffer<PathCell> path)
        {
            int cx = math.clamp(npc.Cell.x / DotsConst.CellSize, 0, DotsConst.CellsPerRow - 1);
            int cy = math.clamp(npc.Cell.y / DotsConst.CellSize, 0, DotsConst.CellsPerRow - 1);

            for (int r = 1; r <= DotsConst.CellsPerRow; r++)
            {
                int count = VisitCandidates(cx, cy, r, npc.Last, -1, out _);
                if (count == 0) continue;
                VisitCandidates(cx, cy, r, npc.Last, npc.Rng.NextInt(count), out Entity picked);
                npc.Target = picked;
                npc.TargetCell = Buildings[picked].Cell;
                npc.State = NpcState.Move;
                ClearPath(ref npc, path);
                return;
            }

            npc.State = NpcState.Wait; // 地图上已无可去的建筑
            npc.WaitTimer = 1f;
        }

        /// <summary>pick &lt; 0 时只计数;否则返回第 pick 个有效候选。</summary>
        int VisitCandidates(int cx, int cy, int r, Entity exclude, int pick, out Entity picked)
        {
            picked = Entity.Null;
            int minX = math.max(cx - r, 0), maxX = math.min(cx + r, DotsConst.CellsPerRow - 1);
            int minY = math.max(cy - r, 0), maxY = math.min(cy + r, DotsConst.CellsPerRow - 1);
            int n = 0;
            for (int y = minY; y <= maxY; y++)
            {
                for (int x = minX; x <= maxX; x++)
                {
                    if (!BuildingMap.TryGetFirstValue(DotsConst.CellHash(x, y), out Entity e, out var it)) continue;
                    do
                    {
                        if (e == exclude || !IsAliveBuilding(e)) continue;
                        if (n == pick)
                        {
                            picked = e;
                            return n + 1;
                        }
                        n++;
                    }
                    while (BuildingMap.TryGetNextValue(out e, ref it));
                }
            }
            return n;
        }

        // ---------- 移动:有绕行路径则沿路径走,否则直线朝目标 ----------

        void Step(Entity self, ref NpcData npc, ref LocalTransform lt, DynamicBuffer<PathCell> path)
        {
            float2 pos = lt.Position.xy;
            bool onPath = npc.PathIndex < path.Length;
            float2 goal = onPath ? DotsConst.Center(path[npc.PathIndex].Cell) : DotsConst.Center(npc.TargetCell);
            float2 to = goal - pos;
            float dist = math.length(to);
            float step = npc.Speed * math.min(Dt, 0.05f); // 限制单帧步长,低帧率时也不会跨格穿墙
            bool reached = dist <= step;
            float2 next = reached ? goal : pos + to / dist * step;

            int2 nextCell = (int2)math.floor(next);
            if (!nextCell.Equals(npc.Cell))
            {
                if (!DotsConst.InMap(nextCell))
                {
                    Blocked(ref npc, path, nextCell, Entity.Null);
                    return;
                }
                int ni = DotsConst.CellIndex(nextCell);
                Entity occ = Occupancy[ni];
                if (occ != Entity.Null)
                {
                    Blocked(ref npc, path, nextCell, occ);
                    return;
                }
                Occupancy[DotsConst.CellIndex(npc.Cell)] = Entity.Null;
                Occupancy[ni] = self;
                npc.Cell = nextCell;
            }

            lt.Position = new float3(next.x, next.y, lt.Position.z);

            if (onPath && reached)
            {
                npc.PathIndex++;
                if (npc.PathIndex >= path.Length) ClearPath(ref npc, path);
            }
        }

        /// <summary>前方被挡:先在半径5格内找可到达的更近点,找不到则攻击前方障碍。</summary>
        void Blocked(ref NpcData npc, DynamicBuffer<PathCell> path, int2 frontCell, Entity front)
        {
            if (npc.RethinkTimer > 0f) return;

            if (FindDetour(npc.Cell, npc.TargetCell, path))
            {
                npc.PathIndex = 0;
                return;
            }

            ClearPath(ref npc, path);
            npc.RethinkTimer = DotsConst.AttackInterval;
            if (front == Entity.Null) return; // 地图边界,无可攻击对象

            npc.ShakeTimer = DotsConst.ShakeTime;
            if (Buildings.HasComponent(front))
            {
                var b = Buildings[front];
                b.Hp -= DotsConst.AttackDamage;
                Buildings[front] = b;
                if (b.Hp <= 0) Kill(front, frontCell);
            }
            else if (Health.HasComponent(front))
            {
                var h = Health[front];
                h.Hp -= DotsConst.AttackDamage;
                Health[front] = h;
                if (h.Hp <= 0) Kill(front, frontCell);
            }
        }

        void Kill(Entity e, int2 cell)
        {
            Occupancy[DotsConst.CellIndex(cell)] = Entity.Null;
            Ecb.DestroyEntity(e);
        }

        bool Free(int2 c) => DotsConst.InMap(c) && Occupancy[DotsConst.CellIndex(c)] == Entity.Null;

        /// <summary>
        /// 以当前格为圆心、半径5格的圆内做 8 方向 BFS(不允许斜穿墙角),
        /// 路径上每一格都在圆内;取离目标最近且比当前格更近的可达格,写入绕行路径。
        /// </summary>
        bool FindDetour(int2 start, int2 goal, DynamicBuffer<PathCell> path)
        {
            const int R = DotsConst.DetourRadius;
            const int S = DotsConst.DetourSpan;

            for (int i = 0; i < S * S; i++) Parent[i] = -1;
            int origin = R * S + R;
            Parent[origin] = origin;
            int head = 0, tail = 0;
            Queue[tail++] = origin;

            int best = -1;
            int2 g0 = start - goal;
            int bestD = g0.x * g0.x + g0.y * g0.y;

            while (head < tail)
            {
                int cur = Queue[head++];
                int lx = cur % S, ly = cur / S;
                int2 curCell = start + new int2(lx - R, ly - R);

                for (int dy = -1; dy <= 1; dy++)
                {
                    for (int dx = -1; dx <= 1; dx++)
                    {
                        if (dx == 0 && dy == 0) continue;
                        int nx = lx + dx, ny = ly + dy;
                        if (nx < 0 || ny < 0 || nx >= S || ny >= S) continue;
                        int ox = nx - R, oy = ny - R;
                        if (ox * ox + oy * oy > R * R) continue;
                        int ni = ny * S + nx;
                        if (Parent[ni] != -1) continue;

                        int2 w = start + new int2(ox, oy);
                        if (!Free(w)) continue;
                        if (dx != 0 && dy != 0 &&
                            (!Free(curCell + new int2(dx, 0)) || !Free(curCell + new int2(0, dy))))
                            continue;

                        Parent[ni] = cur;
                        Queue[tail++] = ni;

                        int2 g = w - goal;
                        int dSq = g.x * g.x + g.y * g.y;
                        if (dSq < bestD)
                        {
                            bestD = dSq;
                            best = ni;
                        }
                    }
                }
            }

            if (best < 0) return false;

            int len = 0;
            for (int c = best; c != origin; c = Parent[c]) len++;
            path.ResizeUninitialized(len);
            int k = len - 1;
            for (int c = best; c != origin; c = Parent[c])
                path[k--] = new PathCell { Cell = start + new int2(c % S - R, c / S - R) };
            return true;
        }
    }

    // ========================= 系统:玩家移动 =========================

    [BurstCompile]
    public partial struct PlayerMoveSystem : ISystem
    {
        [BurstCompile]
        public void OnCreate(ref SystemState state)
        {
            state.RequireForUpdate<PlayerInputData>();
            state.RequireForUpdate<PlayerTag>();
        }

        [BurstCompile]
        public void OnUpdate(ref SystemState state)
        {
            var input = SystemAPI.GetSingleton<PlayerInputData>();
            float2 move = input.Move;
            if (math.lengthsq(move) > 1f) move = math.normalize(move);
            float dt = SystemAPI.Time.DeltaTime;

            foreach (var lt in SystemAPI.Query<RefRW<LocalTransform>>().WithAll<PlayerTag>())
            {
                float3 p = lt.ValueRO.Position;
                p.xy = math.clamp(p.xy + move * input.Speed * dt, 0f, DotsConst.MapSize);
                lt.ValueRW.Position = p;
            }
        }
    }

    // ========================= 渲染辅助 Job =========================

    /// <summary>建筑按食物档位分桶写入矩阵,LevelStart[i]..LevelStart[i+1] 为第 i 档。</summary>
    [BurstCompile]
    public struct BuildingMatricesJob : IJob
    {
        [ReadOnly] public NativeArray<BuildingData> Buildings;
        public NativeArray<float4x4> Matrices;
        public NativeArray<int> LevelStart;

        public void Execute()
        {
            const int L = DotsConst.FoodLevels;
            var cursor = new NativeArray<int>(L, Allocator.Temp);
            for (int i = 0; i < Buildings.Length; i++) cursor[Level(Buildings[i].Food)]++;

            LevelStart[0] = 0;
            for (int l = 0; l < L; l++)
            {
                LevelStart[l + 1] = LevelStart[l] + cursor[l];
                cursor[l] = LevelStart[l];
            }

            for (int i = 0; i < Buildings.Length; i++)
            {
                var b = Buildings[i];
                int idx = cursor[Level(b.Food)]++;
                float2 c = DotsConst.Center(b.Cell);
                Matrices[idx] = float4x4.Translate(new float3(c.x, c.y, 5f));
            }
        }

        static int Level(int food) =>
            math.clamp(food * (DotsConst.FoodLevels - 1) / DotsConst.FoodCap, 0, DotsConst.FoodLevels - 1);
    }

    /// <summary>AI 矩阵:攻击中的 AI 左右晃动。</summary>
    [BurstCompile]
    public struct NpcMatricesJob : IJobParallelFor
    {
        [ReadOnly] public NativeArray<LocalTransform> Transforms;
        [ReadOnly] public NativeArray<NpcData> Npcs;
        public NativeArray<float4x4> Matrices;

        public void Execute(int i)
        {
            float3 p = Transforms[i].Position;
            float s = Npcs[i].ShakeTimer;
            if (s > 0f) p.x += math.sin(s * 60f) * 0.18f;
            Matrices[i] = float4x4.Translate(p);
        }
    }

    // ========================= 主 MonoBehaviour(挂摄像机) =========================

    [RequireComponent(typeof(Camera))]
    [DisallowMultipleComponent]
    public class DOTS : MonoBehaviour
    {
        [Header("素材(留空则 Resources.Load<SpriteAtlas>(\"SpriteAtlas\") 兜底)")]
        public SpriteAtlas Atlas;

        [Header("数量")]
        public int FactoryCount = 10000;
        public int NpcCount = 10000;

        [Header("玩家/摄像机")]
        public float PlayerSpeed = 15f;
        public float MinZoom = 5f;
        public float MaxZoom = 120f;
        public uint Seed = 12345;

        const int ChunkSize = 50;                                  // 草地合并块边长(格)
        const int ChunkRows = DotsConst.MapSize / ChunkSize;       // 20 -> 共 400 块

        Camera _cam;
        EntityManager _em;
        Entity _inputEntity;
        Entity _gridEntity;
        EntityQuery _buildingQuery;
        EntityQuery _npcQuery;
        EntityQuery _playerQuery;

        Mesh _grassChunkMesh;
        Mesh _factoryMesh;
        Mesh _charMesh;
        RenderParams _grassRp;
        RenderParams[] _buildingRps;
        RenderParams _charRp;

        NativeArray<Entity> _occupancy;
        NativeParallelMultiHashMap<int, Entity> _buildingMap;
        NativeArray<float4x4> _buildingMatrices;
        NativeArray<int> _levelStart;
        NativeArray<float4x4> _npcMatrices;
        Matrix4x4[] _chunkMatrices;
        Matrix4x4[] _playerMatrix;

        PerfMonitor _monitor;
        bool _ready;

        // ---------- 初始化 ----------

        void Start()
        {
            var atlas = Atlas != null ? Atlas : Resources.Load<SpriteAtlas>("SpriteAtlas");
            if (atlas == null)
            {
                Debug.LogError("DOTS: 找不到名为 SpriteAtlas 的图集(请挂到 Atlas 字段或放到 Resources 目录)。");
                enabled = false;
                return;
            }

            Sprite grass = atlas.GetSprite("草地");
            Sprite factory = atlas.GetSprite("工厂");
            Sprite character = atlas.GetSprite("角色");
            if (grass == null || factory == null || character == null)
            {
                Debug.LogError("DOTS: 图集缺少精灵(需要:草地 / 工厂 / 角色)。");
                enabled = false;
                return;
            }

            var world = Unity.Entities.World.DefaultGameObjectInjectionWorld;
            if (world == null)
            {
                Debug.LogError("DOTS: 默认 ECS World 不存在,请确认已安装 com.unity.entities。");
                enabled = false;
                return;
            }
            _em = world.EntityManager;

            BuildRendering(grass, factory, character);
            SpawnEntities();
            SetupCamera();

            _monitor = new PerfMonitor(_cam);
            StartCoroutine(_monitor.CaptureLoop());

            _ready = true;
        }

        void BuildRendering(Sprite grass, Sprite factory, Sprite character)
        {
            // 注意:URP 2D Renderer 下 RenderMeshInstanced 只有 URP/Unlit 着色器能正常渲染
            // (Unity 官方 Issue 17371:2D/Sprite 系着色器配合实例化绘制不出画面)。
            var shader = Shader.Find("Universal Render Pipeline/Unlit");
            bool urpUnlit = shader != null;
            if (shader == null) shader = Shader.Find("Sprites/Default");
            if (shader == null) shader = Shader.Find("Unlit/Transparent");

            // alphaClip=true 时启用 Alpha 裁剪(镂空精灵用);草地为不透明整块,无需裁剪。
            Material MakeMat(Sprite s, bool alphaClip, Color tint)
            {
                var m = new Material(shader) { mainTexture = s.texture, enableInstancing = true };
                if (m.HasProperty("_BaseColor")) m.SetColor("_BaseColor", tint);
                if (m.HasProperty("_Color")) m.SetColor("_Color", tint);
                if (urpUnlit && alphaClip)
                {
                    m.SetFloat("_AlphaClip", 1f);
                    m.SetFloat("_Cutoff", 0.5f);
                    m.EnableKeyword("_ALPHATEST_ON");
                    m.renderQueue = (int)RenderQueue.AlphaTest;
                }
                return m;
            }

            var bounds = new Bounds(
                new Vector3(DotsConst.MapSize * 0.5f, DotsConst.MapSize * 0.5f, 0f),
                new Vector3(DotsConst.MapSize * 3f, DotsConst.MapSize * 3f, 100f));

            RenderParams MakeRp(Material m) => new RenderParams(m)
            {
                worldBounds = bounds,
                shadowCastingMode = ShadowCastingMode.Off,
                receiveShadows = false
            };

            _grassRp = MakeRp(MakeMat(grass, false, Color.white));
            _charRp = MakeRp(MakeMat(character, true, Color.white));

            // 食物越多颜色越深:每档一个材质,按档位分批实例化
            _buildingRps = new RenderParams[DotsConst.FoodLevels];
            var darkest = new Color(0.22f, 0.16f, 0.12f, 1f);
            for (int l = 0; l < DotsConst.FoodLevels; l++)
            {
                float t = l / (float)(DotsConst.FoodLevels - 1);
                _buildingRps[l] = MakeRp(MakeMat(factory, true, Color.Lerp(Color.white, darkest, t)));
            }

            _grassChunkMesh = BuildGrassChunkMesh(grass);
            _factoryMesh = BuildUnitQuad(factory);
            _charMesh = BuildUnitQuad(character);

            _buildingMatrices = new NativeArray<float4x4>(FactoryCount, Allocator.Persistent);
            _levelStart = new NativeArray<int>(DotsConst.FoodLevels + 1, Allocator.Persistent);
            _npcMatrices = new NativeArray<float4x4>(NpcCount, Allocator.Persistent);
            _chunkMatrices = new Matrix4x4[ChunkRows * ChunkRows];
            _playerMatrix = new Matrix4x4[1];
        }

        void SpawnEntities()
        {
            var rng = new Random(Seed == 0 ? 1u : Seed);
            int maxPlace = DotsConst.MapSize * DotsConst.MapSize / 2;
            FactoryCount = Mathf.Clamp(FactoryCount, 0, maxPlace);
            NpcCount = Mathf.Clamp(NpcCount, 0, maxPlace);

            _occupancy = new NativeArray<Entity>(DotsConst.MapSize * DotsConst.MapSize, Allocator.Persistent);

            int2 RandomFreeCell(ref Random r)
            {
                while (true)
                {
                    var c = new int2(r.NextInt(0, DotsConst.MapSize), r.NextInt(0, DotsConst.MapSize));
                    if (_occupancy[DotsConst.CellIndex(c)] == Entity.Null) return c;
                }
            }

            // ---- 建筑 + 空间哈希 ----
            _buildingMap = new NativeParallelMultiHashMap<int, Entity>(FactoryCount, Allocator.Persistent);
            var buildingArch = _em.CreateArchetype(typeof(BuildingData));
            using (var ents = _em.CreateEntity(buildingArch, FactoryCount, Allocator.Temp))
            {
                for (int i = 0; i < ents.Length; i++)
                {
                    int2 c = RandomFreeCell(ref rng);
                    _em.SetComponentData(ents[i], new BuildingData
                    {
                        Food = 0,
                        Timer = rng.NextFloat(0f, DotsConst.FoodInterval),
                        Hp = DotsConst.BuildingHp,
                        Cell = c
                    });
                    _occupancy[DotsConst.CellIndex(c)] = ents[i];
                    _buildingMap.Add(DotsConst.BucketHash(c), ents[i]);
                }
            }

            // ---- AI ----
            var npcArch = _em.CreateArchetype(typeof(NpcData), typeof(NpcHealth), typeof(LocalTransform), typeof(PathCell));
            using (var ents = _em.CreateEntity(npcArch, NpcCount, Allocator.Temp))
            {
                for (int i = 0; i < ents.Length; i++)
                {
                    int2 c = RandomFreeCell(ref rng);
                    _em.SetComponentData(ents[i], new NpcData
                    {
                        Cell = c,
                        State = NpcState.Wait,
                        WaitTimer = rng.NextFloat(0f, 3f), // 错峰启动
                        Speed = rng.NextFloat(4f, 8f),
                        Rng = new Random(rng.NextUInt(1u, uint.MaxValue))
                    });
                    _em.SetComponentData(ents[i], new NpcHealth { Hp = DotsConst.NpcHp });
                    float2 p = DotsConst.Center(c);
                    _em.SetComponentData(ents[i], LocalTransform.FromPosition(p.x, p.y, 2f));
                    _occupancy[DotsConst.CellIndex(c)] = ents[i];
                }
            }

            // ---- 玩家(不参与占格) ----
            var player = _em.CreateEntity(typeof(PlayerTag), typeof(LocalTransform));
            _em.SetComponentData(player, LocalTransform.FromPosition(DotsConst.MapSize * 0.5f, DotsConst.MapSize * 0.5f, 1f));

            // ---- 单例 ----
            _inputEntity = _em.CreateEntity(typeof(PlayerInputData));
            _em.SetComponentData(_inputEntity, new PlayerInputData { Move = float2.zero, Speed = PlayerSpeed });

            _gridEntity = _em.CreateEntity(typeof(GridSingleton));
            _em.SetComponentData(_gridEntity, new GridSingleton { Occupancy = _occupancy, BuildingMap = _buildingMap });

            // ---- 渲染查询 ----
            _buildingQuery = _em.CreateEntityQuery(typeof(BuildingData));
            _npcQuery = _em.CreateEntityQuery(typeof(NpcData), typeof(LocalTransform));
            _playerQuery = _em.CreateEntityQuery(typeof(PlayerTag), typeof(LocalTransform));
        }

        void SetupCamera()
        {
            _cam = GetComponent<Camera>();
            _cam.orthographic = true;
            _cam.orthographicSize = 20f;
            transform.position = new Vector3(DotsConst.MapSize * 0.5f, DotsConst.MapSize * 0.5f, -10f);
        }

        // ---------- 每帧 ----------

        void Update()
        {
            if (!_ready) return;

            // 玩家输入写入单例(ECS 系统在本帧 MonoBehaviour.Update 之后执行)
            _em.SetComponentData(_inputEntity, new PlayerInputData { Move = ReadMoveInput(), Speed = PlayerSpeed });
            _monitor.Tick(Time.unscaledDeltaTime);
        }

        void LateUpdate()
        {
            if (!_ready) return;

            // 滚轮缩放
            float scroll = ReadScrollInput();
            if (scroll != 0f)
                _cam.orthographicSize = Mathf.Clamp(_cam.orthographicSize * (1f - scroll * 0.1f), MinZoom, MaxZoom);

            // 摄像机锁定玩家:与本帧提交渲染的玩家位置完全一致,玩家始终位于屏幕绝对中心
            using (var arr = _playerQuery.ToComponentDataArray<LocalTransform>(Allocator.Temp))
            {
                if (arr.Length > 0)
                {
                    float3 p = arr[0].Position;
                    transform.position = new Vector3(p.x, p.y, -10f);
                    _playerMatrix[0] = Matrix4x4.Translate(new Vector3(p.x, p.y, p.z));
                    _monitor.PlayerRenderPos = new Vector3(p.x, p.y, p.z);
                    _monitor.HasPlayer = true;
                    Graphics.RenderMeshInstanced(_charRp, _charMesh, 0, _playerMatrix, 1);
                }
            }

            // 渲染放在 LateUpdate:此时 ECS 已推进完本帧,摄像机与所有实体使用同一帧的数据
            RenderGrass();
            _monitor.BuildingCount = RenderBuildings();
            _monitor.NpcCount = RenderNpcs();
            _monitor.TileCount = DotsConst.MapSize * DotsConst.MapSize;
        }

        // ---------- 输入读取(兼容新版 Input System 与旧版 Input Manager) ----------

        static float2 ReadMoveInput()
        {
#if ENABLE_INPUT_SYSTEM
            var kb = Keyboard.current;
            if (kb == null) return float2.zero;
            return new float2(
                (kb.dKey.isPressed ? 1f : 0f) - (kb.aKey.isPressed ? 1f : 0f),
                (kb.wKey.isPressed ? 1f : 0f) - (kb.sKey.isPressed ? 1f : 0f));
#else
            return new float2(
                (Input.GetKey(KeyCode.D) ? 1f : 0f) - (Input.GetKey(KeyCode.A) ? 1f : 0f),
                (Input.GetKey(KeyCode.W) ? 1f : 0f) - (Input.GetKey(KeyCode.S) ? 1f : 0f));
#endif
        }

        static float ReadScrollInput()
        {
#if ENABLE_INPUT_SYSTEM
            var mouse = Mouse.current;
            if (mouse == null) return 0f;
            // Windows 上滚轮一格通常为 ±120,归一化到 ±1
            return Mathf.Clamp(mouse.scroll.ReadValue().y, -1f, 1f);
#else
            return Input.mouseScrollDelta.y;
#endif
        }

        // ---------- 渲染实现 ----------

        /// <summary>草地:只画视口内的合并块(每块 50x50 格,一次实例化调用)。</summary>
        void RenderGrass()
        {
            float halfH = _cam.orthographicSize + ChunkSize;
            float halfW = halfH * _cam.aspect + ChunkSize;
            Vector3 c = transform.position;

            int minX = Mathf.Clamp(Mathf.FloorToInt((c.x - halfW) / ChunkSize), 0, ChunkRows - 1);
            int maxX = Mathf.Clamp(Mathf.FloorToInt((c.x + halfW) / ChunkSize), 0, ChunkRows - 1);
            int minY = Mathf.Clamp(Mathf.FloorToInt((c.y - halfH) / ChunkSize), 0, ChunkRows - 1);
            int maxY = Mathf.Clamp(Mathf.FloorToInt((c.y + halfH) / ChunkSize), 0, ChunkRows - 1);

            int n = 0;
            for (int y = minY; y <= maxY; y++)
                for (int x = minX; x <= maxX; x++)
                    _chunkMatrices[n++] = Matrix4x4.Translate(new Vector3(x * ChunkSize, y * ChunkSize, 10f));

            if (n > 0)
                Graphics.RenderMeshInstanced(_grassRp, _grassChunkMesh, 0, _chunkMatrices, n);
        }

        int RenderBuildings()
        {
            var buildings = _buildingQuery.ToComponentDataArray<BuildingData>(Allocator.TempJob);
            int count = math.min(buildings.Length, _buildingMatrices.Length);
            if (count > 0)
            {
                new BuildingMatricesJob
                {
                    Buildings = buildings.GetSubArray(0, count),
                    Matrices = _buildingMatrices,
                    LevelStart = _levelStart
                }.Schedule().Complete();

                var m4 = _buildingMatrices.Reinterpret<Matrix4x4>();
                for (int l = 0; l < DotsConst.FoodLevels; l++)
                {
                    int start = _levelStart[l], end = _levelStart[l + 1];
                    for (int i = start; i < end; i += 1023)
                        Graphics.RenderMeshInstanced(_buildingRps[l], _factoryMesh, 0, m4, math.min(1023, end - i), i);
                }
            }
            buildings.Dispose();
            return count;
        }

        int RenderNpcs()
        {
            var transforms = _npcQuery.ToComponentDataArray<LocalTransform>(Allocator.TempJob);
            var npcs = _npcQuery.ToComponentDataArray<NpcData>(Allocator.TempJob);
            int count = math.min(transforms.Length, _npcMatrices.Length);
            if (count > 0)
            {
                new NpcMatricesJob { Transforms = transforms, Npcs = npcs, Matrices = _npcMatrices }
                    .Schedule(count, 256).Complete();

                var m4 = _npcMatrices.Reinterpret<Matrix4x4>();
                for (int i = 0; i < count; i += 1023)
                    Graphics.RenderMeshInstanced(_charRp, _charMesh, 0, m4, math.min(1023, count - i), i);
            }
            transforms.Dispose();
            npcs.Dispose();
            return count;
        }

        // ---------- 网格构建 ----------

        static void GetSpriteUV(Sprite s, out Vector2 uvMin, out Vector2 uvMax)
        {
            var uvs = s.uv;
            uvMin = uvs[0];
            uvMax = uvs[0];
            for (int i = 1; i < uvs.Length; i++)
            {
                uvMin = Vector2.Min(uvMin, uvs[i]);
                uvMax = Vector2.Max(uvMax, uvs[i]);
            }
        }

        /// <summary>1x1 居中四边形(建筑/角色)。</summary>
        static Mesh BuildUnitQuad(Sprite s)
        {
            GetSpriteUV(s, out var mn, out var mx);
            var mesh = new Mesh
            {
                vertices = new[]
                {
                    new Vector3(-0.5f, -0.5f, 0f), new Vector3(0.5f, -0.5f, 0f),
                    new Vector3(-0.5f, 0.5f, 0f), new Vector3(0.5f, 0.5f, 0f)
                },
                uv = new[]
                {
                    new Vector2(mn.x, mn.y), new Vector2(mx.x, mn.y),
                    new Vector2(mn.x, mx.y), new Vector2(mx.x, mx.y)
                },
                colors32 = new[]
                {
                    new Color32(255, 255, 255, 255), new Color32(255, 255, 255, 255),
                    new Color32(255, 255, 255, 255), new Color32(255, 255, 255, 255)
                },
                triangles = new[] { 0, 2, 1, 2, 3, 1 }
            };
            mesh.RecalculateBounds();
            return mesh;
        }

        /// <summary>50x50 格的草地合并块网格(2500 个四边形,复用同一实例矩阵平铺全图)。</summary>
        static Mesh BuildGrassChunkMesh(Sprite s)
        {
            GetSpriteUV(s, out var mn, out var mx);
            int quadCount = ChunkSize * ChunkSize;
            var verts = new Vector3[quadCount * 4];
            var uvs = new Vector2[quadCount * 4];
            var cols = new Color32[quadCount * 4];
            var tris = new int[quadCount * 6];
            for (int i = 0; i < cols.Length; i++) cols[i] = new Color32(255, 255, 255, 255);

            int vi = 0, ti = 0;
            for (int y = 0; y < ChunkSize; y++)
            {
                for (int x = 0; x < ChunkSize; x++)
                {
                    verts[vi] = new Vector3(x, y, 0f);
                    verts[vi + 1] = new Vector3(x + 1, y, 0f);
                    verts[vi + 2] = new Vector3(x, y + 1, 0f);
                    verts[vi + 3] = new Vector3(x + 1, y + 1, 0f);

                    uvs[vi] = new Vector2(mn.x, mn.y);
                    uvs[vi + 1] = new Vector2(mx.x, mn.y);
                    uvs[vi + 2] = new Vector2(mn.x, mx.y);
                    uvs[vi + 3] = new Vector2(mx.x, mx.y);

                    tris[ti] = vi; tris[ti + 1] = vi + 2; tris[ti + 2] = vi + 1;
                    tris[ti + 3] = vi + 2; tris[ti + 4] = vi + 3; tris[ti + 5] = vi + 1;

                    vi += 4;
                    ti += 6;
                }
            }

            var mesh = new Mesh { indexFormat = IndexFormat.UInt16 };
            mesh.vertices = verts;
            mesh.uv = uvs;
            mesh.colors32 = cols;
            mesh.triangles = tris;
            mesh.RecalculateBounds();
            return mesh;
        }

        // ---------- 清理 ----------

        void OnDestroy()
        {
            _monitor?.Dispose();

            var world = Unity.Entities.World.DefaultGameObjectInjectionWorld;
            if (_ready && world != null && world.IsCreated)
            {
                _em.CompleteAllTrackedJobs();
                if (_em.Exists(_gridEntity)) _em.DestroyEntity(_gridEntity); // 先撤单例,AI 系统随即停止访问网格
            }
            if (_occupancy.IsCreated) _occupancy.Dispose();
            if (_buildingMap.IsCreated) _buildingMap.Dispose();
            if (_buildingMatrices.IsCreated) _buildingMatrices.Dispose();
            if (_levelStart.IsCreated) _levelStart.Dispose();
            if (_npcMatrices.IsCreated) _npcMatrices.Dispose();
        }
    }

    // ========================= 性能监测 =========================

    /// <summary>
    /// 帧率/移动延迟/缩放延迟 三条实时曲线 + 统计面板。
    /// 延迟 = 输入事件的系统时间戳 → 首个"屏幕四边像素普遍变化"的帧的出帧时刻(帧末抓屏 + 异步回读四条边)。
    /// 跳帧 = 摄像机开始渲染时,玩家渲染位置投影到屏幕后不在绝对中心(误差 > 0.5 像素)。
    /// </summary>
    public sealed class PerfMonitor
    {
        const int EdgeInset = 2;                 // 采样线距屏幕边缘的像素
        const int PixelDiffThreshold = 12;       // 单像素任一通道变化 > 12/255 视为变化
        const float BroadChangeRatio = 0.5f;     // 四边过半像素变化视为"普遍变化"
        const double LatencyTimeout = 1.0;       // 超时未见画面变化(如贴边/缩放到极限)则丢弃
        const double QuietBeforeInput = 0.1;     // 输入前画面需静止 0.1 秒,避免上一次运动的画面被误判
        const float JumpTolerancePx = 0.5f;
        const int WarmupFrames = 30;
        const float HistBinSec = 0.0001f;        // 帧时间直方图 0.1ms 一档
        const int HistBins = 20000;              // 覆盖 0~2 秒

        sealed class EdgeFrame
        {
            public readonly AsyncGPUReadbackRequest[] Req = new AsyncGPUReadbackRequest[4];
            public double Time;
        }

        readonly Camera _cam;
        readonly Queue<EdgeFrame> _pendingFrames = new Queue<EdgeFrame>();
        readonly Color32[][] _prevEdges = new Color32[4][];
        RenderTexture _rt;
        bool _hasPrevEdges;
        bool _alive = true;
        double _lastBroadChangeTime = -1;

        public Vector3 PlayerRenderPos;
        public bool HasPlayer;
        public int BuildingCount, NpcCount, TileCount;

        // 帧率统计
        long _frames;
        double _totalTime;
        double _fpsMean, _fpsM2;
        readonly int[] _histCount = new int[HistBins];
        readonly double[] _histSum = new double[HistBins];
        int _warmup;

        // 延迟与跳帧
        double _movePendingSince = -1, _zoomPendingSince = -1;
        float _lastMove, _maxMove, _lastZoom, _maxZoom;
        int _moveSamples, _zoomSamples;
        int _jumpCount;
        int _lastJumpCheckFrame = -1;

        // UI
        CurveGraph _fpsGraph, _moveGraph, _zoomGraph;
        Label _stats;
        float _uiTimer;
        readonly StringBuilder _sb = new StringBuilder(512);

#if ENABLE_INPUT_SYSTEM
        readonly System.Action<InputEventPtr, InputDevice> _onInputEvent;
#endif

        public PerfMonitor(Camera cam)
        {
            _cam = cam;
            RenderPipelineManager.beginCameraRendering += OnBeginCameraRendering;
            Camera.onPreRender += OnPreRenderBuiltin;
#if ENABLE_INPUT_SYSTEM
            _onInputEvent = OnInputEvent;
            InputSystem.onEvent += _onInputEvent;
#endif
            ApplyFrameCap(false);
            BuildUI();
        }

        bool _cap60;

        /// <summary>垂直同步会覆盖 targetFrameRate,两种模式下都关闭;锁60只是请求,实际帧率以统计为准。</summary>
        void ApplyFrameCap(bool cap60)
        {
            _cap60 = cap60;
            QualitySettings.vSyncCount = 0;
            Application.targetFrameRate = cap60 ? 60 : -1;
        }

        public void Dispose()
        {
            _alive = false;
            RenderPipelineManager.beginCameraRendering -= OnBeginCameraRendering;
            Camera.onPreRender -= OnPreRenderBuiltin;
#if ENABLE_INPUT_SYSTEM
            InputSystem.onEvent -= _onInputEvent;
#endif
            if (_rt != null)
            {
                _rt.Release();
                Object.Destroy(_rt);
                _rt = null;
            }
        }

        // ---------- 每帧(MonoBehaviour.Update 中调用) ----------

        public void Tick(float dt)
        {
#if ENABLE_INPUT_SYSTEM
            if (Keyboard.current != null && Keyboard.current.rKey.wasPressedThisFrame) ResetStats();
            if (Keyboard.current != null && Keyboard.current.qKey.wasPressedThisFrame)
            {
                ApplyFrameCap(!_cap60);
                ResetStats(); // 两种模式的帧率统计分开,避免混在一起
            }
#endif
            AccumulateFrame(dt);
            PollReadbacks();

            double now = Time.realtimeSinceStartupAsDouble;
            if (_movePendingSince >= 0 && now - _movePendingSince > LatencyTimeout) _movePendingSince = -1;
            if (_zoomPendingSince >= 0 && now - _zoomPendingSince > LatencyTimeout) _zoomPendingSince = -1;

            _uiTimer -= dt;
            if (_uiTimer <= 0f)
            {
                _uiTimer = 0.25f;
                RefreshTexts();
            }
        }

        void AccumulateFrame(float dt)
        {
            if (dt <= 0f) return;
            float fps = 1f / dt;
            _fpsGraph?.Push(fps);
            if (_warmup < WarmupFrames)
            {
                _warmup++;
                return;
            }

            _frames++;
            _totalTime += dt;
            double delta = fps - _fpsMean;
            _fpsMean += delta / _frames;
            _fpsM2 += delta * (fps - _fpsMean);

            int bin = Mathf.Min((int)(dt / HistBinSec), HistBins - 1);
            _histCount[bin]++;
            _histSum[bin] += dt;
        }

        void ResetStats()
        {
            _frames = 0;
            _totalTime = 0;
            _fpsMean = _fpsM2 = 0;
            System.Array.Clear(_histCount, 0, HistBins);
            System.Array.Clear(_histSum, 0, HistBins);
            _maxMove = _maxZoom = _lastMove = _lastZoom = 0f;
            _moveSamples = _zoomSamples = 0;
            _jumpCount = 0;
            _fpsGraph?.Clear();
            _moveGraph?.Clear();
            _zoomGraph?.Clear();
        }

        /// <summary>0.1% Low:最慢 0.1% 帧的平均帧率。</summary>
        float ComputeLow01()
        {
            if (_frames == 0) return 0f;
            double need = System.Math.Max(1.0, _frames * 0.001);
            double cnt = 0, sum = 0;
            for (int b = HistBins - 1; b >= 0 && cnt < need; b--)
            {
                int c = _histCount[b];
                if (c == 0) continue;
                double take = System.Math.Min(c, need - cnt);
                sum += _histSum[b] / c * take;
                cnt += take;
            }
            return sum > 0 ? (float)(cnt / sum) : 0f;
        }

        // ---------- 输入事件:记录按下/滚动的系统时间戳 ----------

#if ENABLE_INPUT_SYSTEM
        void OnInputEvent(InputEventPtr e, InputDevice device)
        {
            if (!e.IsA<StateEvent>() && !e.IsA<DeltaStateEvent>()) return;
            if (_movePendingSince >= 0 || _zoomPendingSince >= 0) return;
            if (_lastBroadChangeTime >= 0 && e.time - _lastBroadChangeTime < QuietBeforeInput) return;

            if (device is Keyboard kb)
            {
                // onEvent 在状态写入前触发:isPressed 仍是旧状态,只在"从静止到按下"时开始计时
                if (MoveHeld(kb)) return;
                if (Pressed(kb.wKey, e) || Pressed(kb.aKey, e) || Pressed(kb.sKey, e) || Pressed(kb.dKey, e))
                    _movePendingSince = e.time;
            }
            else if (device is Mouse mouse)
            {
                var kbNow = Keyboard.current;
                if (kbNow != null && MoveHeld(kbNow)) return;
                if (mouse.scroll.y.ReadValueFromEvent(e, out float v) && v != 0f)
                    _zoomPendingSince = e.time;
            }
        }

        static bool MoveHeld(Keyboard kb) => kb.wKey.isPressed || kb.aKey.isPressed || kb.sKey.isPressed || kb.dKey.isPressed;

        static bool Pressed(ButtonControl key, InputEventPtr e) => key.ReadValueFromEvent(e, out float v) && v >= 0.5f;
#endif

        // ---------- 帧末抓屏 + 异步回读四条边 ----------

        public IEnumerator CaptureLoop()
        {
            var eof = new WaitForEndOfFrame();
            while (_alive)
            {
                yield return eof;
                if (_alive) Capture();
            }
        }

        void Capture()
        {
            int w = Screen.width, h = Screen.height;
            if (w < 16 || h < 16 || !SystemInfo.supportsAsyncGPUReadback) return;

            if (_rt == null || _rt.width != w || _rt.height != h)
            {
                if (_rt != null)
                {
                    _rt.Release();
                    Object.Destroy(_rt);
                }
                _rt = new RenderTexture(w, h, 0, RenderTextureFormat.ARGB32);
                _rt.Create();
                _hasPrevEdges = false;
            }

            ScreenCapture.CaptureScreenshotIntoRenderTexture(_rt);

            var f = new EdgeFrame { Time = Time.realtimeSinceStartupAsDouble };
            int i = EdgeInset;
            f.Req[0] = AsyncGPUReadback.Request(_rt, 0, 0, w, i, 1, 0, 1, TextureFormat.RGBA32);          // 下边
            f.Req[1] = AsyncGPUReadback.Request(_rt, 0, 0, w, h - 1 - i, 1, 0, 1, TextureFormat.RGBA32);  // 上边
            f.Req[2] = AsyncGPUReadback.Request(_rt, 0, i, 1, 0, h, 0, 1, TextureFormat.RGBA32);          // 左边
            f.Req[3] = AsyncGPUReadback.Request(_rt, 0, w - 1 - i, 1, 0, h, 0, 1, TextureFormat.RGBA32);  // 右边
            _pendingFrames.Enqueue(f);
        }

        void PollReadbacks()
        {
            while (_pendingFrames.Count > 0)
            {
                var f = _pendingFrames.Peek();
                bool done = true, error = false;
                for (int k = 0; k < 4; k++)
                {
                    if (!f.Req[k].done) done = false;
                    else if (f.Req[k].hasError) error = true;
                }
                if (!done) break;

                _pendingFrames.Dequeue();
                if (error)
                {
                    _hasPrevEdges = false;
                    continue;
                }
                ProcessEdges(f);
            }
        }

        void ProcessEdges(EdgeFrame f)
        {
            bool comparable = _hasPrevEdges;
            int total = 0, changed = 0;

            for (int k = 0; k < 4; k++)
            {
                var data = f.Req[k].GetData<Color32>();
                var prev = _prevEdges[k];
                if (prev == null || prev.Length != data.Length)
                {
                    prev = _prevEdges[k] = new Color32[data.Length];
                    comparable = false;
                }
                if (comparable)
                {
                    for (int p = 0; p < data.Length; p++)
                    {
                        Color32 a = data[p], b = prev[p];
                        int diff = Mathf.Max(Mathf.Abs(a.r - b.r), Mathf.Max(Mathf.Abs(a.g - b.g), Mathf.Abs(a.b - b.b)));
                        if (diff > PixelDiffThreshold) changed++;
                    }
                    total += data.Length;
                }
                data.CopyTo(prev);
            }
            _hasPrevEdges = true;

            if (total == 0 || changed < total * BroadChangeRatio) return;

            _lastBroadChangeTime = f.Time;
            if (_movePendingSince >= 0 && f.Time > _movePendingSince)
            {
                _lastMove = (float)((f.Time - _movePendingSince) * 1000.0);
                _maxMove = Mathf.Max(_maxMove, _lastMove);
                _moveSamples++;
                _moveGraph?.Push(_lastMove);
                _movePendingSince = -1;
            }
            if (_zoomPendingSince >= 0 && f.Time > _zoomPendingSince)
            {
                _lastZoom = (float)((f.Time - _zoomPendingSince) * 1000.0);
                _maxZoom = Mathf.Max(_maxZoom, _lastZoom);
                _zoomSamples++;
                _zoomGraph?.Push(_lastZoom);
                _zoomPendingSince = -1;
            }
        }

        // ---------- 跳帧检测:摄像机真正开始渲染时检查玩家是否在屏幕绝对中心 ----------

        void OnBeginCameraRendering(ScriptableRenderContext ctx, Camera cam) => CheckJump(cam);

        void OnPreRenderBuiltin(Camera cam) => CheckJump(cam);

        void CheckJump(Camera cam)
        {
            if (cam != _cam || !HasPlayer || _warmup < WarmupFrames) return;
            if (_lastJumpCheckFrame == Time.frameCount) return;
            _lastJumpCheckFrame = Time.frameCount;

            Vector3 sp = cam.WorldToScreenPoint(PlayerRenderPos);
            float dx = sp.x - cam.pixelWidth * 0.5f;
            float dy = sp.y - cam.pixelHeight * 0.5f;
            if (Mathf.Abs(dx) > JumpTolerancePx || Mathf.Abs(dy) > JumpTolerancePx) _jumpCount++;
        }

        // ---------- UI ----------

        void BuildUI()
        {
            var doc = Object.FindFirstObjectByType<UIDocument>();
            if (doc == null)
            {
                var go = new GameObject("DOTS 性能监测 UI");
                doc = go.AddComponent<UIDocument>();
                doc.panelSettings = ScriptableObject.CreateInstance<PanelSettings>();
            }
            var root = doc.rootVisualElement;
            if (root == null) return;

            // 面板距屏幕边缘留出空隙,保证四边采样线不被 UI 覆盖
            var panel = new VisualElement { pickingMode = PickingMode.Ignore };
            panel.style.position = Position.Absolute;
            panel.style.left = 40;
            panel.style.top = 40;
            panel.style.flexDirection = FlexDirection.Column;

            _fpsGraph = new CurveGraph(300, new Color(0.35f, 1f, 0.45f, 0.95f));
            _moveGraph = new CurveGraph(60, new Color(0.4f, 0.75f, 1f, 0.95f));
            _zoomGraph = new CurveGraph(60, new Color(1f, 0.75f, 0.3f, 0.95f));

            _stats = new Label { pickingMode = PickingMode.Ignore };
            _stats.style.width = 300;
            _stats.style.backgroundColor = new Color(0f, 0f, 0f, 0.45f);
            _stats.style.color = new Color(1f, 1f, 1f, 0.92f);
            _stats.style.fontSize = 12;
            _stats.style.paddingLeft = _stats.style.paddingRight = 6;
            _stats.style.paddingTop = _stats.style.paddingBottom = 4;
            SetRadius(_stats.style, 4);
            _stats.style.whiteSpace = WhiteSpace.Normal;

            panel.Add(_fpsGraph);
            panel.Add(_moveGraph);
            panel.Add(_zoomGraph);
            panel.Add(_stats);
            root.Add(panel);
        }

        static void SetRadius(IStyle s, float r)
        {
            s.borderTopLeftRadius = s.borderTopRightRadius = s.borderBottomLeftRadius = s.borderBottomRightRadius = r;
        }

        void RefreshTexts()
        {
            if (_stats == null) return;

            _fpsGraph.SetTitle($"帧率  当前 {_fpsGraph.Latest:F1}  (纵轴上限 {_fpsGraph.ScaleMax:F0})");
            _moveGraph.SetTitle($"移动延迟  最近 {_lastMove:F1} ms  最高 {_maxMove:F1} ms");
            _zoomGraph.SetTitle($"缩放延迟  最近 {_lastZoom:F1} ms  最高 {_maxZoom:F1} ms");

            float avg = _totalTime > 0 ? (float)(_frames / _totalTime) : 0f;
            float std = _frames > 1 ? (float)System.Math.Sqrt(_fpsM2 / (_frames - 1)) : 0f;

            _sb.Clear();
            _sb.Append("地块数 ").Append(TileCount.ToString("N0"))
               .Append("    建筑数 ").Append(BuildingCount.ToString("N0"))
               .Append("    角色数 ").Append(NpcCount.ToString("N0")).Append('\n');
            _sb.Append("平均帧率 ").Append(avg.ToString("F1"))
               .Append("    帧率标准差 ").Append(std.ToString("F1"))
               .Append("    0.1%Low ").Append(ComputeLow01().ToString("F1")).Append('\n');
            _sb.Append("摄像机跳帧次数 ").Append(_jumpCount).Append('\n');
            _sb.Append("移动最高延迟 ").Append(_maxMove.ToString("F1")).Append(" ms (").Append(_moveSamples).Append(" 次)")
               .Append("    缩放最高延迟 ").Append(_maxZoom.ToString("F1")).Append(" ms (").Append(_zoomSamples).Append(" 次)\n");
            _sb.Append("帧率模式 ").Append(_cap60 ? "锁定60" : "无上限").Append('\n');
            _sb.Append("[Q] 切换帧率模式    [R] 重置统计");
            _stats.text = _sb.ToString();
        }

        /// <summary>半透明背景 + 折线的环形缓冲曲线。</summary>
        sealed class CurveGraph : VisualElement
        {
            readonly float[] _values;
            readonly Color _color;
            readonly Label _title;
            readonly VisualElement _plot;
            int _head, _count;

            public float Latest { get; private set; }
            public float ScaleMax { get; private set; } = 1f;

            public CurveGraph(int capacity, Color color)
            {
                _values = new float[capacity];
                _color = color;
                pickingMode = PickingMode.Ignore;
                style.width = 300;
                style.height = 92;
                style.marginBottom = 6;
                style.paddingLeft = style.paddingRight = 6;
                style.paddingTop = style.paddingBottom = 4;
                style.backgroundColor = new Color(0f, 0f, 0f, 0.45f);
                SetRadius(style, 4);

                _title = new Label { pickingMode = PickingMode.Ignore };
                _title.style.color = new Color(1f, 1f, 1f, 0.92f);
                _title.style.fontSize = 12;

                _plot = new VisualElement { pickingMode = PickingMode.Ignore };
                _plot.style.flexGrow = 1;
                _plot.generateVisualContent += Draw;

                Add(_title);
                Add(_plot);
            }

            public void SetTitle(string text) => _title.text = text;

            public void Push(float v)
            {
                _values[_head] = v;
                _head = (_head + 1) % _values.Length;
                if (_count < _values.Length) _count++;
                Latest = v;
                _plot.MarkDirtyRepaint();
            }

            public void Clear()
            {
                _head = _count = 0;
                Latest = 0f;
                _plot.MarkDirtyRepaint();
            }

            float At(int i) => _values[(_head - _count + i + _values.Length) % _values.Length];

            void Draw(MeshGenerationContext mgc)
            {
                var r = _plot.contentRect;
                if (r.width <= 1f || r.height <= 1f) return;

                float max = 1f;
                for (int i = 0; i < _count; i++) max = Mathf.Max(max, At(i));
                ScaleMax = max * 1.15f;

                var p = mgc.painter2D;
                p.lineWidth = 1f;
                p.strokeColor = new Color(1f, 1f, 1f, 0.15f);
                p.BeginPath();
                p.MoveTo(new Vector2(0f, r.height * 0.5f));
                p.LineTo(new Vector2(r.width, r.height * 0.5f));
                p.Stroke();

                if (_count < 2) return;
                p.lineWidth = 1.5f;
                p.strokeColor = _color;
                p.BeginPath();
                float stepX = r.width / (_values.Length - 1);
                for (int i = 0; i < _count; i++)
                {
                    var pt = new Vector2(i * stepX, r.height - Mathf.Clamp01(At(i) / ScaleMax) * r.height);
                    if (i == 0) p.MoveTo(pt);
                    else p.LineTo(pt);
                }
                p.Stroke();
            }
        }
    }
}
