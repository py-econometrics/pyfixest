export const onRequest: PagesFunction = async (context) => {
  const accept = context.request.headers.get('Accept') || '';
  if (!accept.includes('text/markdown')) {
    return context.next();
  }
  const url = new URL(context.request.url);
  const pathname = url.pathname;
  const candidates: string[] = [];
  if (pathname.endsWith('.html')) {
    candidates.push(pathname + '.md');
  } else if (pathname.endsWith('/')) {
    candidates.push(pathname + 'index.html.md');
  } else if (!pathname.split('/').pop()?.includes('.')) {
    candidates.push(pathname + '.html.md', pathname + '/index.html.md');
  } else {
    return context.next();
  }

  for (const mdPath of candidates) {
    const mdUrl = new URL(mdPath, url.origin);
    const mdResponse = await context.env.ASSETS.fetch(mdUrl);
    if (mdResponse.ok) {
      return new Response(mdResponse.body, {
        headers: { 'Content-Type': 'text/markdown; charset=utf-8' },
      });
    }
  }
  return context.next();
};
