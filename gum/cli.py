from dotenv import load_dotenv
load_dotenv()

import argparse
import asyncio
from gum import gum
from gum.observers import Screen

def parse_args():
    parser = argparse.ArgumentParser(description='GUM - A Python package with command-line interface')
    parser.add_argument('--user-name', '-u', type=str, help='The user name to use')
    parser.add_argument('--query', '-q', type=str, help='Query to run')
    parser.add_argument('--limit', '-l', type=int, help='Limit the number of results', default=10)
    parser.add_argument('--model', '-m', type=str, help='Model to use')
    return parser.parse_args()

async def main():
    args = parse_args()
    print(f"User Name: {args.user_name}")
    
    if args.query is not None:
        gum_instance = gum(args.user_name, args.model)
        await gum_instance.connect_db()
        result = await gum_instance.query(args.query, limit=10)
        
        # pretty print confidences / propositions / number of items returned
        print(f"\nFound {len(result)} results:")
        for prop, score in result:
            print(f"\nProposition: {prop.text}")
            if prop.reasoning:
                print(f"Reasoning: {prop.reasoning}")
            if prop.confidence is not None:
                print(f"Confidence: {prop.confidence:.2f}")
            print(f"Relevance Score: {score:.2f}")
            print("-" * 80)
        
    else:
        async with gum(args.user_name, args.model, Screen(args.model)) as gum_instance:
            await asyncio.Future()  # run forever (Ctrl-C to stop)

def cli():
    asyncio.run(main())

if __name__ == '__main__':
    cli()