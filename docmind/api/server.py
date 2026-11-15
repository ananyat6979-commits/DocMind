"""API entry point with socket-peer identity preserved for IP quotas."""
import argparse


def main():
    import uvicorn
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host',default='127.0.0.1');parser.add_argument('--port',type=int,default=8000);parser.add_argument('--workers',type=int,default=1)
    args=parser.parse_args()
    if not 1<=args.port<=65535 or not 1<=args.workers<=16:parser.error('Port or worker count outside supported bounds')
    uvicorn.run('docmind.api.main:app',host=args.host,port=args.port,workers=args.workers,proxy_headers=False)


if __name__=='__main__':main()
