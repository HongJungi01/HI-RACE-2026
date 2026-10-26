-- 공유 차고 테이블 (Supabase SQL Editor에서 한 번 실행)
create table if not exists public.garage_cars (
  id          uuid primary key default gen_random_uuid(),
  owner       uuid not null default auth.uid() references auth.users(id) on delete cascade,
  owner_name  text not null default '',
  name        text not null check (char_length(name) between 1 and 40),
  note        text not null default '' check (char_length(note) <= 200),
  design      jsonb not null,
  code        text not null default '' check (char_length(code) <= 60000),
  updated_at  timestamptz not null default now(),
  unique (owner, name)
);

alter table public.garage_cars enable row level security;

-- 누구나(로그인 안 해도) 목록 읽기
create policy "garage read" on public.garage_cars for select using (true);
-- 로그인한 사람은 자기 차만 추가·수정·삭제
create policy "garage insert own" on public.garage_cars for insert to authenticated with check (owner = auth.uid());
create policy "garage update own" on public.garage_cars for update to authenticated using (owner = auth.uid()) with check (owner = auth.uid());
create policy "garage delete own" on public.garage_cars for delete to authenticated using (owner = auth.uid());

-- 설계(JSON) 크기 제한 약 200 KB
alter table public.garage_cars add constraint design_size check (pg_column_size(design) < 200000);

-- 다른 사람이 올리면 목록이 바로 갱신되게 (Realtime)
alter publication supabase_realtime add table public.garage_cars;
