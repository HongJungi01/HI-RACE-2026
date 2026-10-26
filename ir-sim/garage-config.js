// 공유 차고 서버 설정 (Supabase). GitHub Pages 등 claude.ai 밖에서 쓸 때만 사용.
// Supabase 프로젝트 Settings → API의 Project URL과 anon public 키를 넣으세요.
// anon 키는 공개돼도 되는 키입니다 (권한은 테이블의 RLS 정책이 막음). service_role 키는 절대 넣지 마세요.
window.GARAGE_CONFIG = {
  url: "https://tndzcdfbpkzibtnifmdc.supabase.co",
  anonKey: "",   // 예: "eyJhbGciOi..."
};
