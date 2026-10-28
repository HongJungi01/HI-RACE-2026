-- 주행 로그 테이블 (Supabase SQL Editor에서 한 번 실행, garage_setup.sql 다음)
-- 탭 2 '로그 녹화' → 맵 + 차마다 위치·방향 기록을 올리고, 탭 3 '로그 시청'에서 재생
create table if not exists public.sim_logs (
  id          uuid primary key default gen_random_uuid(),
  owner       uuid not null default auth.uid() references auth.users(id) on delete cascade,
  owner_name  text not null default '',
  meta        jsonb not null,                                        -- 맵·길이·차 이름/색 (목록용)
  data        text not null check (char_length(data) <= 2000000),   -- 위치 기록 (약 10분 · 4대 ≈ 0.8 MB)
  created_at  timestamptz not null default now()
);

alter table public.sim_logs enable row level security;

-- 누구나(로그인 안 해도) 목록·재생
create policy "simlog read" on public.sim_logs for select using (true);
-- 로그인한 사람은 자기 기록만 올리기·삭제
create policy "simlog insert own" on public.sim_logs for insert to authenticated with check (owner = auth.uid());
create policy "simlog delete own" on public.sim_logs for delete to authenticated using (owner = auth.uid());

alter table public.sim_logs add constraint simlog_meta_size check (pg_column_size(meta) < 20000);
create index if not exists sim_logs_created on public.sim_logs (created_at desc);

-- 다른 사람이 올리면 목록이 바로 갱신되게 (Realtime)
alter publication supabase_realtime add table public.sim_logs;
