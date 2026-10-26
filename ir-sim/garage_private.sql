-- 공유 차고: 코드 비공개 기능 (garage_setup.sql 다음에 한 번 실행)
-- 비공개로 올리면 garage_cars.code는 비우고 컴파일된 펌웨어(hex)만 공개,
-- 원본 코드는 올린 사람만 읽을 수 있는 garage_code 테이블에 보관

alter table public.garage_cars
  add column if not exists code_private boolean not null default false,
  add column if not exists hex text check (hex is null or char_length(hex) <= 400000);

-- 비공개면 공개 코드 칸은 반드시 비어 있어야 함
alter table public.garage_cars drop constraint if exists private_code_empty;
alter table public.garage_cars add constraint private_code_empty check (not code_private or code = '');

create table if not exists public.garage_code (
  car_id      uuid primary key references public.garage_cars(id) on delete cascade,
  owner       uuid not null default auth.uid() references auth.users(id) on delete cascade,
  code        text not null check (char_length(code) <= 60000),
  updated_at  timestamptz not null default now()
);

alter table public.garage_code enable row level security;

-- 본인 것만 읽기·쓰기 (다른 사람은 행이 없는 것처럼 보임)
create policy "code select own" on public.garage_code for select to authenticated using (owner = auth.uid());
create policy "code insert own" on public.garage_code for insert to authenticated
  with check (owner = auth.uid() and exists (select 1 from public.garage_cars c where c.id = car_id and c.owner = auth.uid()));
create policy "code update own" on public.garage_code for update to authenticated using (owner = auth.uid()) with check (owner = auth.uid());
create policy "code delete own" on public.garage_code for delete to authenticated using (owner = auth.uid());
