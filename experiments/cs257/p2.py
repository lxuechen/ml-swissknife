import fire


def implies(p, q):
    return not p or q


def part_b(P, Q, R, S):
    return implies(
        implies(implies((P and Q), R), S),
        implies(implies(implies(R, Q), P), S)
    )


def main(part="c"):
    if part == "b":
        out = []
        for P in (True, False):
            for Q in (True, False):
                for R in (True, False):
                    for S in (True, False):
                        res = part_b(P, Q, R, S)
                        out.append(res)
                        if not res:
                            print(P, Q, R, S)
        print(out)
    elif part == "d":
        def get_rgb(idx):  # get rgb variable index from node index
            return (idx - 1) * 3 + 1, (idx - 1) * 3 + 2, (idx - 1) * 3 + 3

        out = []
        # vertex cons
        for node_id in (1, 2, 3, 4):
            # get rgb assignment
            r, g, b = get_rgb(node_id)
            out.append(f"{r} {g} {b} 0\n")
            out.append(f"{-r} {-g} 0\n")
            out.append(f"{-r} {-b} 0\n")
            out.append(f"{-g} {-b} 0\n")

        # edge cons
        for edge in (
            (1, 2), (1, 3),
            (2, 3), (2, 4),
            (3, 4)
        ):
            ri, gi, bi = get_rgb(edge[0])
            rj, gj, bj = get_rgb(edge[1])
            out.append(f"{-ri} {-rj} 0\n")
            out.append(f"{-gi} {-gj} 0\n")
            out.append(f"{-bi} {-bj} 0\n")

        print(f'num vars={12}, num clauses={len(out)}')
        print(''.join(out))
    elif part == "c":
        for A in (True, False):
            for B in (True, False):
                print(
                    not (implies(implies(A, B),
                                 implies(not B, not A)))
                )


if __name__ == "__main__":
    fire.Fire(main)
